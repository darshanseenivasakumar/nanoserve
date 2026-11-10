"""M5: greedy speculative decoding with a small draft model.

Idea (Leviathan et al. 2023; Chen et al. 2023): decode is memory-bound, so a forward pass over k+1 tokens
costs the target about the same as over 1 token. Let a cheap DRAFT model guess k tokens, then let the
TARGET check all of them in ONE forward pass:

    tokens so far: ... x                          (target KV holds everything except x)
    draft (k cheap steps):      d1 d2 d3 d4
    target, one pass over [x d1 d2 d3 d4] -> its own greedy choices  t0 t1 t2 t3 t4
                                (t_i = what the target would emit after x, d1..d_i)
    accept the longest prefix where d_{i+1} == t_i, then add the target's token at the first mismatch:
        d1==t0, d2==t1, d3!=t2  ->  emit d1 d2 t2   (3 tokens for one target pass)
        all 4 match             ->  emit d1 d2 d3 d4 t4  (the "bonus" token)

Because every emitted token is literally the target's greedy choice given the tokens before it, the
output is IDENTICAL to plain greedy decoding with the target (in exact arithmetic). The draft only
changes speed: tokens per target pass = accepted + 1.

Cache bookkeeping (both models use the paged pool; blocks are reserved up front per request):
  * target KV always covers every token except the newest; after verification it is REWOUND to drop the
    rejected drafts.
  * the draft may lag by one extra token (when all k drafts are accepted it never saw d_k or the bonus);
    its next round starts with a multi-token "catch-up" step (masked attention on an existing context).

Shortcuts (flagged): blocks for prompt + max_new + k + 1 are reserved at admission (no preemption here);
attention uses the gather+SDPA path for both models (not the M4 kernel); fixed k (no adaptive lookahead).
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import torch

from nanoserve.engine import StepOutput
from nanoserve.kv_cache import BlockTable, PagedKVCache
from nanoserve.model import Qwen2Model, SeqInput


@dataclass(eq=False)
class SpecSeq:
    seq_id: int
    prompt_ids: list[int]
    max_new_tokens: int
    ignore_eos: bool
    arrival_time: float
    output_ids: list[int] = field(default_factory=list)
    t_table: BlockTable | None = None
    d_table: BlockTable | None = None

    @property
    def tokens(self) -> list[int]:
        return self.prompt_ids + self.output_ids


class SpeculativeEngine:
    def __init__(self, target: Qwen2Model, draft: Qwen2Model, eos_token_ids: list[int],
                 num_speculative_tokens: int = 4, max_seq_len: int = 2048, max_num_seqs: int = 8,
                 max_num_batched_tokens: int = 4096, target_kv_tokens: int | None = None,
                 draft_kv_tokens: int | None = None, block_size: int = 16):
        if target.cfg.vocab_size != draft.cfg.vocab_size:
            raise ValueError("draft and target must share a vocabulary")
        self.target, self.draft = target, draft
        self.k = num_speculative_tokens
        self.eos_token_ids = list(eos_token_ids)
        self._eos = set(eos_token_ids)
        self.max_seq_len = max_seq_len
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max(max_num_batched_tokens, max_seq_len)
        self.block_size = block_size
        t_tokens = target_kv_tokens or 8 * (max_seq_len + self.k + 1)
        d_tokens = draft_kv_tokens or t_tokens
        self.tpool = PagedKVCache.from_token_budget(target.cfg, t_tokens, block_size, target.device, target.dtype)
        self.dpool = PagedKVCache.from_token_budget(draft.cfg, d_tokens, block_size, draft.device, draft.dtype)
        self.waiting: deque[SpecSeq] = deque()
        self.running: list[SpecSeq] = []
        self.sequences: dict[int, SpecSeq] = {}
        self._next_id = 0
        self.reset_stats()

    # ---- bookkeeping -----------------------------------------------------------------------------------
    def reset_stats(self) -> None:
        self._stats = {"steps": 0, "target_forwards": 0, "draft_forwards": 0, "proposed": 0, "accepted": 0,
                       "emitted_by_verification": 0, "peak_running": 0}

    @property
    def stats(self) -> dict:
        s = dict(self._stats)
        s["acceptance_rate"] = s["accepted"] / s["proposed"] if s["proposed"] else None
        s["tokens_per_target_forward"] = (s["emitted_by_verification"] / s["target_forwards"]
                                          if s["target_forwards"] else None)
        s["num_speculative_tokens"] = self.k
        return s

    def _reserve_tokens(self, seq: SpecSeq) -> int:
        return len(seq.prompt_ids) + seq.max_new_tokens + self.k + 1

    def add_request(self, prompt_ids: list[int], max_new_tokens: int, ignore_eos: bool = False,
                    arrival_time: float | None = None) -> int:
        seq = SpecSeq(self._next_id, list(prompt_ids), max_new_tokens, ignore_eos,
                      time.perf_counter() if arrival_time is None else arrival_time)
        if len(prompt_ids) + max_new_tokens > self.max_seq_len:
            raise ValueError(f"prompt + max_new_tokens exceeds max_seq_len ({self.max_seq_len})")
        need = math.ceil(self._reserve_tokens(seq) / self.block_size)
        if need > min(self.tpool.num_blocks, self.dpool.num_blocks):
            raise ValueError("request can never fit in the KV pools")
        self.waiting.append(seq)
        self.sequences[seq.seq_id] = seq
        self._next_id += 1
        return seq.seq_id

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def _release(self, seq: SpecSeq) -> None:
        if seq.t_table is not None:
            seq.t_table.release(self.tpool.allocator)
            seq.d_table.release(self.dpool.allocator)
            seq.t_table = seq.d_table = None
        self.sequences.pop(seq.seq_id, None)

    def abort(self, seq_id: int) -> None:
        seq = self.sequences.get(seq_id)
        if seq is None:
            return
        if seq in self.running:
            self.running.remove(seq)
        elif seq in self.waiting:
            self.waiting.remove(seq)
        self._release(seq)

    def _admit(self) -> list[SpecSeq]:
        admitted, budget = [], self.max_num_batched_tokens
        while self.waiting and len(self.running) + len(admitted) < self.max_num_seqs:
            seq = self.waiting[0]
            need = math.ceil(self._reserve_tokens(seq) / self.block_size)
            if (len(seq.prompt_ids) > budget or need > self.tpool.allocator.num_free
                    or need > self.dpool.allocator.num_free):
                break
            self.waiting.popleft()
            seq.t_table, seq.d_table = BlockTable(self.block_size), BlockTable(self.block_size)
            seq.t_table.reserve(self._reserve_tokens(seq), self.tpool.allocator)
            seq.d_table.reserve(self._reserve_tokens(seq), self.dpool.allocator)
            budget -= len(seq.prompt_ids)
            admitted.append(seq)
        self.running.extend(admitted)
        return admitted

    # ---- one batched forward over (table, tokens, rows-wanted) items --------------------------------
    def _forward(self, model: Qwen2Model, pool: PagedKVCache, items) -> torch.Tensor | None:
        """items: list of (table, new_tokens, n_rows). Feeds new_tokens on top of each table's context,
        writes their K/V, returns logits for the LAST n_rows tokens of each item (concatenated)."""
        tokens, positions, inputs, slots, rows, offset = [], [], [], [], [], 0
        for table, toks, n_rows in items:
            T, ctx = len(toks), table.num_tokens
            slots += table.slots(T)
            inputs.append(SeqInput(T, block_table=torch.tensor(table.blocks, device=model.device) if ctx else None,
                                   ctx_len=ctx))
            tokens += toks
            positions += range(ctx, ctx + T)
            rows += range(offset + T - n_rows, offset + T)
            offset += T
        dev = model.device
        logits = model.forward_batch(torch.tensor(tokens, device=dev), torch.tensor(positions, device=dev), inputs,
                                     logits=torch.tensor(rows, device=dev, dtype=torch.long), paged=pool,
                                     slot_mapping=torch.tensor(slots, device=dev))
        for table, toks, _ in items:
            table.commit(len(toks))
        return logits

    # ---- one engine step ---------------------------------------------------------------------------------
    def step(self) -> list[StepOutput]:
        admitted = self._admit()
        decoding = [s for s in self.running if s not in admitted]
        if not admitted and not decoding:
            return []
        st = self._stats
        st["steps"] += 1
        st["peak_running"] = max(st["peak_running"], len(self.running))

        # 1) DRAFT. Prefill new requests; for running ones, catch up on unseen tokens and propose d1,
        #    then k-1 more one-token steps.
        drafts: dict[SpecSeq, list[int]] = {s: [] for s in decoding}
        items = [(s.d_table, s.prompt_ids, 0) for s in admitted]
        items += [(s.d_table, s.tokens[s.d_table.num_tokens:], 1) for s in decoding]
        if items:
            logits = self._forward(self.draft, self.dpool, items)
            st["draft_forwards"] += 1
            for s, tok in zip(decoding, logits.argmax(-1).tolist()):
                drafts[s].append(tok)
        for _ in range(self.k - 1):
            if not decoding:
                break
            logits = self._forward(self.draft, self.dpool, [(s.d_table, [drafts[s][-1]], 1) for s in decoding])
            st["draft_forwards"] += 1
            for s, tok in zip(decoding, logits.argmax(-1).tolist()):
                drafts[s].append(tok)

        # 2) TARGET. Prefill new requests (1 row: their first token); verify running ones: feed
        #    [newest token, d1..dk] and read the target's greedy choice at all k+1 positions.
        items = [(s.t_table, s.prompt_ids, 1) for s in admitted]
        items += [(s.t_table, s.tokens[s.t_table.num_tokens:] + drafts[s], self.k + 1) for s in decoding]
        choices = self._forward(self.target, self.tpool, items).argmax(-1).tolist()
        st["target_forwards"] += 1

        # 3) ACCEPT and emit.
        outputs, pos = [], 0
        for s in admitted:
            outputs += self._emit(s, [choices[pos]])
            pos += 1
        for s in decoding:
            t = choices[pos:pos + self.k + 1]
            pos += self.k + 1
            d = drafts[s]
            n = 0
            while n < self.k and d[n] == t[n]:
                n += 1
            st["proposed"] += self.k
            st["accepted"] += n
            new_tokens = d[:n] + [t[n]]
            st["emitted_by_verification"] += len(new_tokens)
            outputs += self._emit(s, new_tokens)
        return outputs

    def _emit(self, s: SpecSeq, new_tokens: list[int]) -> list[StepOutput]:
        out, finished = [], False
        for tok in new_tokens:
            s.output_ids.append(tok)
            finished = (tok in self._eos and not s.ignore_eos) or len(s.output_ids) >= s.max_new_tokens
            out.append(StepOutput(s.seq_id, tok, finished))
            if finished:
                break
        if finished:
            self.running.remove(s)
            self._release(s)
            return out
        # Rewind: KV is valid only for tokens that were actually accepted. Every token except the newest
        # must be cached; anything cached beyond that belongs to rejected drafts.
        keep = len(s.tokens) - 1
        s.t_table.rewind(min(s.t_table.num_tokens, keep))
        s.d_table.rewind(min(s.d_table.num_tokens, keep))
        return out
