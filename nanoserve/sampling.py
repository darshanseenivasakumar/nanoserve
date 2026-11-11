"""M13: sampling (temperature, top-k, top-p) with per-request, batch-independent randomness.

Greedy decoding always picks the highest-scoring token. Sampling picks token j with probability
softmax(logits / temperature)_j, optionally restricted to the k best tokens (top-k) or to the smallest set
of best tokens whose probabilities add up to p (top-p, "nucleus"). Low temperature = close to greedy;
high = more random.

How we draw, per row (the Gumbel-max trick):
    token = argmax_j ( x_j + g_j ),   g_j = -log(-log(u_j)),   u_j ~ Uniform(0, 1)
is an exact sample from softmax(x). It needs no cumulative sums and works on the masked logits directly.

Randomness per request: the u's for request r's n-th generated token come from a generator seeded with
mix(seed_r, n). So a request's draws don't depend on which other requests share its batch, nor on a
preemption + recompute (same n -> same draw): the same seed gives the same text as long as the logits are
the same.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

_MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0       # 0 = greedy
    top_p: float = 1.0             # 1 = off
    top_k: int = 0                 # 0 = off
    seed: int | None = None        # None = the engine picks one (still per-request reproducible internally)

    def __post_init__(self):
        if not self.temperature >= 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0 or self.top_k == 1


def mix_seed(seed: int, step: int) -> int:
    """splitmix64 of (seed, step): well-spread 64-bit generator seeds for consecutive steps."""
    z = (seed * 0x9E3779B97F4A7C15 + step + 1) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & ((1 << 63) - 1)


def sample(logits: torch.Tensor, params: list[SamplingParams | None], seeds: list[int],
           steps: list[int]) -> list[int]:
    """logits [B, V] -> one token per row. Row i uses params[i] (None = greedy) and draws its randomness
    from (seeds[i], steps[i]). An all-greedy batch is exactly the old argmax."""
    greedy = [p is None or p.greedy for p in params]
    if all(greedy):
        return logits.argmax(-1).tolist()
    out = logits.argmax(-1)                              # correct for the greedy rows
    rows = [i for i, g in enumerate(greedy) if not g]
    x = logits[rows].float()
    R, V = x.shape
    dev = x.device
    x = x / torch.tensor([params[i].temperature for i in rows], device=dev)[:, None]

    ks = [params[i].top_k if params[i].top_k > 0 else V for i in rows]
    if min(ks) < V:                                      # top-k: drop everything below the row's k-th score
        kmax = max(ks)
        top = x.topk(kmax, dim=-1).values
        kth = top.gather(1, torch.tensor([k - 1 for k in ks], device=dev)[:, None])
        x = x.masked_fill(x < kth, float("-inf"))

    ps = torch.tensor([params[i].top_p for i in rows], device=dev)
    if (ps < 1).any():                                   # top-p: smallest best set reaching probability p
        sx, idx = x.sort(dim=-1, descending=True)
        probs = sx.softmax(-1)
        before = probs.cumsum(-1) - probs                # probability mass ranked above each token
        sx = sx.masked_fill(before >= ps[:, None], float("-inf"))   # the top token always survives
        x = torch.full_like(x, float("-inf")).scatter(1, idx, sx)

    u = torch.empty(R, V, device=dev)
    for r, i in enumerate(rows):                         # one small generator per request and step
        g = torch.Generator(device=dev)
        g.manual_seed(mix_seed(seeds[i], steps[i]))
        u[r].uniform_(generator=g)
    gumbel = -torch.log(-torch.log(u.clamp_(min=1e-20, max=1 - 1e-7)))
    out[rows] = (x + gumbel).argmax(-1)
    return out.tolist()
