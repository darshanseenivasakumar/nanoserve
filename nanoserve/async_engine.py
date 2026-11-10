"""Run an LLMEngine in a background thread and stream tokens to asyncio callers.

    caller (asyncio) ── inbox (thread-safe queue) ──▶ engine thread: add_request → step() → step() ...
    caller (asyncio) ◀── its own asyncio.Queue ◀── loop.call_soon_threadsafe(token, timestamp, finished)

Why a thread: engine.step() is blocking GPU work; running it on the asyncio loop would freeze every
caller. Only the engine thread ever touches the LLMEngine, so the engine needs no locks.
Each request remembers ITS event loop, so one engine serves callers from any number of loops
(M1/M0 bugs taught us an asyncio object belongs to exactly one loop).

Timestamps are taken in the engine thread right after step() returns (the batch argmax has already been
copied to the CPU, so those tokens really exist). M6's HTTP server reuses this class.
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections.abc import AsyncIterator

from nanoserve.engine import LLMEngine


class AsyncEngine:
    def __init__(self, engine: LLMEngine, idle_poll_s: float = 0.05):
        self.engine = engine
        self.idle_poll_s = idle_poll_s
        self._inbox: queue.Queue = queue.Queue()
        self._streams: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()

    def _ensure_started(self) -> None:
        with self._start_lock:
            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._thread = threading.Thread(target=self._run, name="nanoserve-engine", daemon=True)
                self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    async def generate(self, prompt_ids: list[int], max_new_tokens: int,
                       ignore_eos: bool = False) -> AsyncIterator[tuple[int, float]]:
        """Yield (token, timestamp) as the engine produces them."""
        self._ensure_started()
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        self._inbox.put((list(prompt_ids), max_new_tokens, ignore_eos, loop, q, time.perf_counter()))
        while True:
            item = await q.get()
            if isinstance(item, BaseException):
                raise item
            token, t, finished = item
            yield token, t
            if finished:
                return

    # ---- engine thread -------------------------------------------------------------------------------
    @staticmethod
    def _send(loop, q, item) -> None:
        try:
            loop.call_soon_threadsafe(q.put_nowait, item)
        except RuntimeError:
            pass  # the caller's loop has closed; nobody is listening any more

    def _admit(self, item) -> None:
        prompt, max_new, ignore_eos, loop, q, t_arrival = item
        try:
            sid = self.engine.add_request(prompt, max_new, ignore_eos=ignore_eos, arrival_time=t_arrival)
        except Exception as e:
            self._send(loop, q, e)
            return
        self._streams[sid] = (loop, q)

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self.engine.has_unfinished():          # idle: block (briefly) for new work
                try:
                    self._admit(self._inbox.get(timeout=self.idle_poll_s))
                except queue.Empty:
                    continue
            while True:                                   # admit everything that has arrived
                try:
                    self._admit(self._inbox.get_nowait())
                except queue.Empty:
                    break
            if not self.engine.has_unfinished():
                continue
            try:
                outputs = self.engine.step()
            except Exception as e:                        # fail every in-flight request, keep serving
                for loop, q in self._streams.values():
                    self._send(loop, q, e)
                self._streams.clear()
                for sid in list(self.engine.sequences):
                    self.engine.abort(sid)                # frees their KV blocks too
                continue
            if not outputs:                               # e.g. static policy waiting for its batch to fill
                time.sleep(0.001)
                continue
            t = time.perf_counter()
            for out in outputs:
                loop, q = self._streams[out.seq_id]
                self._send(loop, q, (out.token, t, out.finished))
                if out.finished:
                    del self._streams[out.seq_id]
