"""M11: run the engine loop in its OWN PROCESS, so the HTTP side can never stall the GPU.

M11 profile (c=32): the bare engine loop keeps the GPU 95% busy and makes 2,150 tok/s; the same engine
behind AsyncEngine (a thread) makes 1,928, and behind the HTTP server 1,681 (-22%). In one process, the
engine thread and the HTTP/asyncio thread share Python's GIL: when a step finishes, the engine thread
must wait for the GIL while the other thread formats SSE chunks and detokenizes, and the GPU sits idle.
vLLM v1 splits API server and engine core into separate processes for the same reason.

    API process (FastAPI, tokenizer)                          engine process (owns the GPU)
      generate() ──("add", rid, prompt_ids, ...)──pipe──▶       drain requests, step(), step(), ...
      reader thread ◀──("tokens", t, [(rid, tok, finished)])──  ONE message per step, not per token
        └─ one call_soon_threadsafe per event loop per step

EngineProcess has AsyncEngine's interface (generate / close / num_aborted / eos_token_ids), so the
server and the benchmark don't care which one they get. Each process has its own GIL; the child is
started with "spawn" (CUDA cannot be used in a forked child). Timestamps are stamped in the child with
time.perf_counter(), which on Linux is CLOCK_MONOTONIC: one clock for every process on the machine.
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import threading
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass
class EngineSpec:
    """Everything the child needs to build the engine (must be picklable: it crosses the process boundary)."""
    model: str
    dtype: str = "bfloat16"
    max_seq_len: int = 2048
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 2048
    kv_cache_tokens: int | None = None
    attn_backend: str | None = None
    cuda_graphs: bool | None = None
    piecewise_graphs: bool | None = None
    fused_ops: bool | None = None
    prefix_caching: bool = False
    draft_model: str | None = None
    num_speculative_tokens: int = 4
    capture_graphs: bool = True        # capture every CUDA graph before serving (server); tests skip it


def build_engine(spec: EngineSpec):
    """LLMEngine (or SpeculativeEngine with a draft model) from a spec. Used by the child process and by the
    server's in-thread mode, so both build exactly the same engine."""
    import torch

    from nanoserve.engine import LLMEngine, load_eos_token_ids
    from nanoserve.model import Qwen2Model

    dtype = getattr(torch, spec.dtype)
    target = Qwen2Model.from_pretrained(spec.model, device="cuda", dtype=dtype)
    eos = load_eos_token_ids(spec.model)
    if spec.draft_model:
        from nanoserve.speculative import SpeculativeEngine
        draft = Qwen2Model.from_pretrained(spec.draft_model, device="cuda", dtype=dtype)
        return SpeculativeEngine(target, draft, eos, num_speculative_tokens=spec.num_speculative_tokens,
                                 max_seq_len=spec.max_seq_len, max_num_seqs=spec.max_num_seqs,
                                 max_num_batched_tokens=spec.max_num_batched_tokens,
                                 target_kv_tokens=spec.kv_cache_tokens)
    return LLMEngine(target, eos, max_seq_len=spec.max_seq_len, max_num_seqs=spec.max_num_seqs,
                     max_num_batched_tokens=spec.max_num_batched_tokens, kv_cache="paged",
                     kv_cache_tokens=spec.kv_cache_tokens, attn_backend=spec.attn_backend,
                     cuda_graphs=spec.cuda_graphs, piecewise_graphs=spec.piecewise_graphs,
                     fused_ops=spec.fused_ops, prefix_caching=spec.prefix_caching)


# ---------------------------------------------------------------------------------------- child process
def _child_stats(llm, num_aborted: int) -> dict:
    st = {"unfinished": llm.has_unfinished(), "num_aborted": num_aborted}
    kv = getattr(llm, "kv", None)
    if kv is not None and hasattr(kv, "num_free_blocks"):
        st["num_free_blocks"] = kv.num_free_blocks
    engine_stats = llm.stats
    st["engine"] = engine_stats() if callable(engine_stats) else engine_stats   # method on SpeculativeEngine
    return st


def _picklable(e: BaseException) -> BaseException:
    """Exceptions cross the pipe pickled; fall back to a RuntimeError carrying the message."""
    import pickle
    try:
        pickle.dumps(e)
        return e
    except Exception:
        return RuntimeError(f"{type(e).__name__}: {e}")


def _child_main(spec: EngineSpec, req_conn, out_conn) -> None:
    try:
        llm = build_engine(spec)
        capture_s = llm.capture_graphs() if (spec.capture_graphs and hasattr(llm, "capture_graphs")) else 0.0
        out_conn.send(("ready", {"eos_token_ids": list(llm.eos_token_ids), "capture_s": capture_s}))
    except BaseException as e:                                  # noqa: BLE001 -- report, then die
        out_conn.send(("fatal", f"{type(e).__name__}: {e}"))
        return
    rid_of: dict[int, int] = {}                                 # engine seq_id -> API request id
    seq_of: dict[int, int] = {}                                 # API request id -> engine seq_id
    num_aborted = 0
    while True:
        # Requests: block briefly when idle (no busy loop), otherwise just drain what has arrived.
        timeout = 0.0 if llm.has_unfinished() else 0.05
        try:
            while req_conn.poll(timeout):
                timeout = 0.0
                msg = req_conn.recv()
                kind = msg[0]
                if kind == "add":
                    _, rid, prompt, max_new, ignore_eos, t_arrival, sampling = msg
                    try:
                        kw = {"sampling": sampling} if sampling is not None else {}
                        sid = llm.add_request(prompt, max_new, ignore_eos=ignore_eos, arrival_time=t_arrival, **kw)
                    except Exception as e:                      # rejected (too long, ...): tell the caller
                        out_conn.send(("error", [rid], _picklable(e)))
                        continue
                    rid_of[sid], seq_of[rid] = rid, sid
                elif kind == "abort":
                    sid = seq_of.pop(msg[1], None)
                    if sid is not None:
                        llm.abort(sid)
                        del rid_of[sid]
                        num_aborted += 1
                elif kind == "stats":
                    out_conn.send(("stats", msg[1], _child_stats(llm, num_aborted)))
                elif kind == "stop":
                    return
        except (EOFError, OSError):                             # the API process is gone
            return
        if not llm.has_unfinished():
            continue
        try:
            outputs = llm.step()
        except Exception as e:                                  # fail every in-flight request, keep serving
            out_conn.send(("error", list(seq_of), _picklable(e)))
            for sid in list(rid_of):
                llm.abort(sid)
            rid_of.clear()
            seq_of.clear()
            continue
        if not outputs:                                         # e.g. static policy waiting for a full batch
            time.sleep(0.001)
            continue
        t = time.perf_counter()
        batch = []
        for out in outputs:
            rid = rid_of[out.seq_id]
            batch.append((rid, out.token, out.finished))
            if out.finished:
                del rid_of[out.seq_id], seq_of[rid]
        out_conn.send(("tokens", t, batch))


# ------------------------------------------------------------------------------------------ API process
class EngineProcess:
    def __init__(self, spec: EngineSpec, start_timeout_s: float = 900.0):
        ctx = mp.get_context("spawn")
        req_r, self._req_w = ctx.Pipe(duplex=False)
        self._out_r, out_w = ctx.Pipe(duplex=False)
        self._proc = ctx.Process(target=_child_main, args=(spec, req_r, out_w), name="nanoserve-engine",
                                 daemon=True)
        self._proc.start()
        req_r.close()                                           # the child holds its own copies
        out_w.close()
        if not self._out_r.poll(start_timeout_s):
            self._proc.kill()
            raise TimeoutError(f"engine process not ready after {start_timeout_s} s")
        kind, info = self._out_r.recv()
        if kind != "ready":
            self._proc.join(timeout=5)
            raise RuntimeError(f"engine process failed to start: {info}")
        self.eos_token_ids: list[int] = info["eos_token_ids"]
        self.capture_s: float = info["capture_s"]
        self._lock = threading.Lock()                           # guards _streams, _next_rid and _req_w
        self._streams: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self._next_rid = 0
        self._stats_waiters: dict[int, list] = {}
        self.num_aborted = 0
        self._reader = threading.Thread(target=self._read_loop, name="nanoserve-engine-reader", daemon=True)
        self._reader.start()

    def is_alive(self) -> bool:
        return self._proc.is_alive()

    def _send(self, msg) -> None:
        with self._lock:
            self._req_w.send(msg)

    async def generate(self, prompt_ids: list[int], max_new_tokens: int, ignore_eos: bool = False,
                       sampling=None) -> AsyncIterator[tuple[int, float]]:
        """Yield (token, timestamp) as the engine process produces them. sampling: SamplingParams or None."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            rid = self._next_rid
            self._next_rid += 1
            self._streams[rid] = (loop, q)
            self._req_w.send(("add", rid, list(prompt_ids), max_new_tokens, ignore_eos, time.perf_counter(),
                              sampling))
        done = False
        try:
            while True:
                item = await q.get()
                if isinstance(item, BaseException):
                    done = True
                    raise item
                token, t, finished = item
                if finished:
                    done = True
                yield token, t
                if finished:
                    return
        finally:
            if not done:                                        # caller stopped listening: abort, free KV
                with self._lock:
                    self._streams.pop(rid, None)
                    self._req_w.send(("abort", rid))
                    self.num_aborted += 1

    def stats(self, timeout_s: float = 30.0) -> dict:
        """Ask the child for its state (free KV blocks, unfinished work, abort count, engine stats)."""
        ev, box = threading.Event(), []
        with self._lock:
            key = self._next_rid
            self._next_rid += 1
            self._stats_waiters[key] = [ev, box]
            self._req_w.send(("stats", key))
        if not ev.wait(timeout_s):
            raise TimeoutError("engine process did not answer")
        return box[0]

    @staticmethod
    def _deliver(items) -> None:
        for q, item in items:
            q.put_nowait(item)

    def _dispatch(self, pairs) -> None:
        """pairs: [(rid, item)]. One call_soon_threadsafe per event loop, not per token."""
        per_loop = defaultdict(list)
        with self._lock:
            for rid, item in pairs:
                stream = self._streams.get(rid)
                if stream is None:                              # aborted meanwhile
                    continue
                loop, q = stream
                per_loop[loop].append((q, item))
                if isinstance(item, BaseException) or item[2]:  # error or finished: stream is done
                    del self._streams[rid]
        for loop, items in per_loop.items():
            try:
                loop.call_soon_threadsafe(self._deliver, items)
            except RuntimeError:
                pass                                            # that caller's loop has closed

    def _read_loop(self) -> None:
        while True:
            try:
                msg = self._out_r.recv()
            except (EOFError, OSError):
                with self._lock:
                    rids = list(self._streams)
                self._dispatch([(rid, RuntimeError("engine process exited")) for rid in rids])
                return
            kind = msg[0]
            if kind == "tokens":
                _, t, batch = msg
                self._dispatch([(rid, (tok, t, fin)) for rid, tok, fin in batch])
            elif kind == "error":
                _, rids, exc = msg
                self._dispatch([(rid, exc) for rid in rids])
            elif kind == "stats":
                _, key, st = msg
                with self._lock:
                    ev, box = self._stats_waiters.pop(key)
                box.append(st)
                ev.set()

    def close(self, timeout_s: float = 30.0) -> None:
        if self._proc.is_alive():
            try:
                self._send(("stop",))
            except (OSError, BrokenPipeError):
                pass
            self._proc.join(timeout=timeout_s)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=5)
