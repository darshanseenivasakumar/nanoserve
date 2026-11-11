"""Qwen2 forward pass written from scratch with plain tensors.

Architecture (per token, per layer), the classic "pre-norm" decoder:

    x ── RMSNorm ── attention (Q,K,V proj + RoPE + GQA) ── o_proj ──(+)── RMSNorm ── SwiGLU MLP ──(+)──▶
    └──────────────────── residual ──────────────────────────────┘└──────── residual ───────────┘

then a final RMSNorm and the LM head (tied to the embedding table) turn the last hidden state into one
score ("logit") per vocabulary token.

The order of floating-point operations deliberately mirrors HuggingFace's modeling_qwen2.py (RMSNorm in
fp32, cos/sin cast to the model dtype, SDPA with enable_gqa, LM head only on the rows we need) so that
bf16 results can match HF bit-for-bit. Change the math here and the token-for-token test will tell you.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    n_layers: int
    n_heads: int            # query heads
    n_kv_heads: int         # key/value heads (GQA: n_heads / n_kv_heads queries share one K/V head)
    head_dim: int
    intermediate_size: int  # MLP width
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool

    @classmethod
    def from_hf_config(cls, d: dict) -> "ModelConfig":
        rope_theta = d.get("rope_theta") or (d.get("rope_parameters") or {}).get("rope_theta")
        return cls(
            vocab_size=d["vocab_size"], hidden_size=d["hidden_size"], n_layers=d["num_hidden_layers"],
            n_heads=d["num_attention_heads"], n_kv_heads=d["num_key_value_heads"],
            head_dim=d.get("head_dim") or d["hidden_size"] // d["num_attention_heads"],
            intermediate_size=d["intermediate_size"], rope_theta=float(rope_theta),
            rms_norm_eps=d["rms_norm_eps"], tie_word_embeddings=d.get("tie_word_embeddings", False),
        )

    @classmethod
    def from_pretrained(cls, name: str) -> "ModelConfig":
        return cls.from_hf_config(json.loads((_snapshot(name) / "config.json").read_text()))

    def kv_bytes_per_token(self, dtype: torch.dtype) -> int:
        """KV-cache memory one token costs: every layer stores one K and one V vector per KV head."""
        return self.n_layers * 2 * self.n_kv_heads * self.head_dim * torch.finfo(dtype).bits // 8


def _snapshot(name: str) -> Path:
    """Local folder holding the model files (downloads config + safetensors once, then uses the cache)."""
    if Path(name).is_dir():
        return Path(name)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(name, allow_patterns=["*.json", "*.safetensors"]))


@dataclass
class LayerWeights:
    ln1: torch.Tensor   # input_layernorm (RMSNorm scale)
    q_w: torch.Tensor
    q_b: torch.Tensor
    k_w: torch.Tensor
    k_b: torch.Tensor
    v_w: torch.Tensor
    v_b: torch.Tensor
    o_w: torch.Tensor
    ln2: torch.Tensor   # post_attention_layernorm
    gate_w: torch.Tensor
    up_w: torch.Tensor
    down_w: torch.Tensor
    # M10: merged matrices (one GEMM instead of three / two), built by merge(). After it, q_w, k_w, v_w (and
    # biases) and gate_w, up_w above are row VIEWS into these: nothing stored twice, unfused path unchanged.
    qkv_w: torch.Tensor | None = None
    qkv_b: torch.Tensor | None = None
    gate_up_w: torch.Tensor | None = None

    HF_NAMES = {
        "ln1": "input_layernorm.weight",
        "q_w": "self_attn.q_proj.weight", "q_b": "self_attn.q_proj.bias",
        "k_w": "self_attn.k_proj.weight", "k_b": "self_attn.k_proj.bias",
        "v_w": "self_attn.v_proj.weight", "v_b": "self_attn.v_proj.bias",
        "o_w": "self_attn.o_proj.weight",
        "ln2": "post_attention_layernorm.weight",
        "gate_w": "mlp.gate_proj.weight", "up_w": "mlp.up_proj.weight", "down_w": "mlp.down_proj.weight",
    }

    @classmethod
    def from_state_dict(cls, sd: dict, i: int) -> "LayerWeights":
        return cls(**{field: sd[f"model.layers.{i}.{hf}"] for field, hf in cls.HF_NAMES.items()})

    def merge(self) -> None:
        """Build the merged matrices and turn the per-matrix fields into views of them. Lazy (only the fused
        path needs it) and one layer at a time: when nothing else holds the originals they are freed as each
        field is replaced, so loading never holds two copies of the weights. (M10 first merged eagerly at
        load time: +50% peak memory, which crashed WSL when the 1.5B fp32 test model loaded.)"""
        if self.qkv_w is not None:
            return
        nq, nk, ng = self.q_w.shape[0], self.k_w.shape[0], self.gate_w.shape[0]
        self.qkv_w = torch.cat([self.q_w, self.k_w, self.v_w])
        self.q_w, self.k_w, self.v_w = self.qkv_w[:nq], self.qkv_w[nq:nq + nk], self.qkv_w[nq + nk:]
        self.qkv_b = torch.cat([self.q_b, self.k_b, self.v_b])
        self.q_b, self.k_b, self.v_b = self.qkv_b[:nq], self.qkv_b[nq:nq + nk], self.qkv_b[nq + nk:]
        self.gate_up_w = torch.cat([self.gate_w, self.up_w])
        self.gate_w, self.up_w = self.gate_up_w[:ng], self.gate_up_w[ng:]


@dataclass
class SeqInput:
    """One sequence's slice of a batched step: how many of the flattened tokens are its, and its KV.

    Contiguous (M1/M2): `cache` = a per-request ContiguousKVCache.
    Paged (M3):         `block_table` = this request's block ids (int tensor), `ctx_len` = tokens already
                        stored before this step; the pool + slot mapping are passed to forward_batch.
    """
    n_tokens: int
    cache: object | None = None
    block_table: torch.Tensor | None = None
    ctx_len: int = 0
    use_kernel: bool = False      # M4: this one-token decode is handled by the Triton paged-attention kernel


@dataclass
class DecodeMeta:
    """M4: built ONCE per step for all kernel-handled decode requests (not once per layer)."""
    token_idx: torch.Tensor       # [Bd] positions of their tokens in the flattened batch
    block_tables: torch.Tensor    # [Bd, max_blocks] int32
    context_lens: torch.Tensor    # [Bd] int32, including this step's token


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Scale x so its root-mean-square is 1, then multiply by a learned per-channel weight.
    Computed in fp32 (bf16 squares overflow/round badly), cast back before the weight multiply -- as HF."""
    dtype = x.dtype
    x = x.to(torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class Qwen2Model:
    """Weights + forward. No nn.Module: nothing is trained, and plain tensors keep the data flow visible."""

    fused = False      # M10: True on the view returned by with_fused_ops()

    def __init__(self, cfg: ModelConfig, state_dict: dict[str, torch.Tensor]):
        self.cfg = cfg
        self.embed = state_dict["model.embed_tokens.weight"]
        self.norm = state_dict["model.norm.weight"]
        self.lm_head = self.embed if cfg.tie_word_embeddings else state_dict["lm_head.weight"]
        self.layers = [LayerWeights.from_state_dict(state_dict, i) for i in range(cfg.n_layers)]
        self.device = self.embed.device
        self.dtype = self.embed.dtype
        # RoPE inverse frequencies, fp32, computed exactly like HF (on CPU, then moved).
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float) / cfg.head_dim))
        self.inv_freq = inv_freq.to(self.device)

    @classmethod
    def from_pretrained(cls, name: str, device: str | torch.device = "cuda",
                        dtype: torch.dtype = torch.bfloat16) -> "Qwen2Model":
        from safetensors.torch import load_file
        folder = _snapshot(name)
        cfg = ModelConfig.from_pretrained(str(folder))
        raw: dict[str, torch.Tensor] = {}
        for f in sorted(folder.glob("*.safetensors")):
            raw.update(load_file(str(f), device=str(device)))
        # Convert one tensor at a time and drop the original at once, so a bf16 -> fp32 load never holds both
        # copies of the model (1.5B on CPU: 3.1 + 6.2 GB at once was too close to WSL's 11 GB cap; M10).
        sd = {k: raw.pop(k).to(dtype) for k in list(raw)}
        return cls(cfg, sd)

    def with_fused_ops(self) -> "Qwen2Model":
        """M10: the same model (shared weights, no copies) whose token-wise ops run as fused Triton kernels:
        one-pass RMSNorm (+ residual add), merged QKV GEMM + RoPE + KV write, merged gate/up GEMM + SiLU*mul.
        The original object stays unfused: it is the path that matches HF bit for bit. (Its per-matrix weights
        become views of the merged ones: same values, same layout, so its results don't change.)"""
        for layer in self.layers:
            layer.merge()
        m = copy.copy(self)
        m.fused = True
        return m

    def final_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            from nanoserve.kernels.fused import rms_norm as fused_rms_norm
            return fused_rms_norm(x, self.norm, self.cfg.rms_norm_eps)
        return rms_norm(x, self.norm, self.cfg.rms_norm_eps)

    def named_weights(self) -> dict[str, torch.Tensor]:
        """Weights under their HF names (used by tests to prove loading is exact)."""
        out = {"model.embed_tokens.weight": self.embed, "model.norm.weight": self.norm}
        if not self.cfg.tie_word_embeddings:
            out["lm_head.weight"] = self.lm_head
        for i, layer in enumerate(self.layers):
            for field, hf in LayerWeights.HF_NAMES.items():
                out[f"model.layers.{i}.{hf}"] = getattr(layer, field)
        return out

    def rope(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin tables for the given token positions, shape [1, T, head_dim], in the model dtype.
        Each pair of channels in Q and K gets rotated by angle position * inv_freq[i]."""
        freqs = positions[None, :, None].float() * self.inv_freq.to(device=positions.device, dtype=torch.float)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    # A layer is split in three so that M9's piecewise CUDA graphs can replay _post + _pre (token-wise work,
    # shapes depend only on the total token count) and run only _attend (per-sequence work) eagerly.
    # forward_batch calls the very same functions, so eager and graphed steps do identical math.

    def _pre(self, i: int, x: torch.Tensor, cos, sin, paged=None, slot_mapping=None):
        """Token-wise first half of layer i: RMSNorm, Q/K/V projections, RoPE, KV write.
        x: [1, N, hidden]. Returns q [1, N, n_heads, hd], k and v [1, N, n_kv_heads, hd]."""
        layer = self.layers[i]
        if self.fused:
            # M10: 3 kernels (norm, ONE QKV GEMM with bias, RoPE + KV write) instead of ~23.
            from nanoserve.kernels.fused import rms_norm as fused_rms_norm, rope_qkv
            cfg = self.cfg
            qkv = F.linear(fused_rms_norm(x, layer.ln1, cfg.rms_norm_eps), layer.qkv_w, layer.qkv_b)[0]
            q, k, v = rope_qkv(qkv, cos[0], sin[0], cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
                               paged.k[i] if paged is not None else None,
                               paged.v[i] if paged is not None else None, slot_mapping)
            return q[None], k[None], v[None]
        h = rms_norm(x, layer.ln1, self.cfg.rms_norm_eps)
        N = h.shape[1]
        hd = self.cfg.head_dim
        q = F.linear(h, layer.q_w, layer.q_b).view(1, N, -1, hd)   # [1, N, n_heads, hd]
        k = F.linear(h, layer.k_w, layer.k_b).view(1, N, -1, hd)   # [1, N, n_kv_heads, hd]
        v = F.linear(h, layer.v_w, layer.v_b).view(1, N, -1, hd)
        c, s = cos.unsqueeze(2), sin.unsqueeze(2)                   # [1, N, 1, hd]: broadcast over heads
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        if paged is not None:
            # Paged (M3): store every token's K/V of this step with ONE indexed copy (slot mapping).
            paged.write(i, k[0], v[0], slot_mapping)
        return q, k, v

    def _post(self, i: int, x: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        """Token-wise second half of layer i: output projection + residual, RMSNorm, SwiGLU MLP + residual."""
        layer = self.layers[i]
        if self.fused:
            # M10: residual add fused into the norm; ONE gate/up GEMM; SiLU*mul in one kernel.
            from nanoserve.kernels.fused import rms_norm as fused_rms_norm, silu_mul
            h, x = fused_rms_norm(F.linear(o, layer.o_w), layer.ln2, self.cfg.rms_norm_eps, residual=x)
            return x + F.linear(silu_mul(F.linear(h, layer.gate_up_w)), layer.down_w)
        x = x + F.linear(o, layer.o_w)
        h = rms_norm(x, layer.ln2, self.cfg.rms_norm_eps)
        return x + F.linear(F.silu(F.linear(h, layer.gate_w)) * F.linear(h, layer.up_w), layer.down_w)

    def _attend(self, i: int, q, k, v, seqs: list[SeqInput], paged=None, decode_meta=None,
                out: torch.Tensor | None = None) -> torch.Tensor:
        """The per-sequence middle of layer i. q/k/v: [1, N, heads, hd] = the flattened tokens of every
        sequence in this step. Returns the attention output [1, N, n_heads * hd] (written into `out` if given).

        SELECTIVE BATCHING: the projections and RoPE (_pre/_post) are token-wise, so they run ONCE over all N
        tokens. Attention mixes tokens *within* a sequence, against that sequence's own cache, so it runs
        once PER SEQUENCE -- except one-token decodes, which M4's kernel handles all together.
        """
        N = q.shape[1]
        hd = self.cfg.head_dim
        o_all = out
        if decode_meta is not None:
            # M4: every one-token decode request of this step in ONE Triton launch, reading K/V in place.
            from nanoserve.kernels.paged_attention import paged_attention_decode
            o_dec = paged_attention_decode(q[0][decode_meta.token_idx], paged.k[i], paged.v[i],
                                           decode_meta.block_tables, decode_meta.context_lens, hd ** -0.5)
            if o_all is None:
                o_all = torch.empty(1, N, q.shape[2] * hd, device=q.device, dtype=q.dtype)
            o_all[0].index_copy_(0, decode_meta.token_idx, o_dec.reshape(o_dec.shape[0], -1))
        outs, start = [], 0
        for seq in seqs:
            T = seq.n_tokens
            if seq.use_kernel:
                start += T
                continue
            qs = q[:, start:start + T].transpose(1, 2)              # [1, n_heads, T, hd]
            ks = k[:, start:start + T].transpose(1, 2)
            vs = v[:, start:start + T].transpose(1, 2)
            mask = None
            if paged is not None and seq.ctx_len > 0:
                # Read this request's whole context (cached + this step's tokens) out of its blocks.
                ks, vs = paged.gather(i, seq.block_table, seq.ctx_len + T)
                if T > 1:
                    # Several new tokens on top of a context (spec-decode verification, M5): query j may
                    # see keys 0 .. ctx_len + j. SDPA's is_causal would align the wrong corner, so build it.
                    L = seq.ctx_len + T
                    mask = torch.arange(L, device=qs.device)[None, :] <= (seq.ctx_len + torch.arange(T, device=qs.device))[:, None]
            elif seq.cache is not None:
                # Contiguous (M1/M2): store this step's K/V, get back K/V for ALL tokens so far.
                ks, vs = seq.cache.write(i, ks, vs)
            if mask is not None:
                # enable_gqa is not supported together with an explicit mask on every SDPA backend, so
                # expand the K/V heads to match the query heads (what HF's repeat_kv does).
                rep = qs.shape[1] // ks.shape[1]
                ks, vs = ks.repeat_interleave(rep, dim=1), vs.repeat_interleave(rep, dim=1)
                o = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=mask, scale=hd ** -0.5)
                outs.append((start, o.transpose(1, 2).contiguous().reshape(1, T, -1).contiguous()))
                start += T
                continue
            # is_causal only for multi-token steps (prefill): token t must not see tokens > t.
            # A single decode token may see its whole cache, so no mask is needed.
            # NOTE: SDPA's is_causal assumes queries and keys start at the same position, so a multi-token
            # step on top of a non-empty CONTIGUOUS cache would be WRONG; forward_batch forbids it.
            o = F.scaled_dot_product_attention(qs, ks, vs, is_causal=T > 1, scale=hd ** -0.5, enable_gqa=True)
            outs.append((start, o.transpose(1, 2).contiguous().reshape(1, T, -1).contiguous()))
            start += T
        if o_all is not None:                     # kernel rows already placed; drop in the SDPA rows
            for s0, o in outs:
                o_all[:, s0:s0 + o.shape[1]] = o
            return o_all
        return outs[0][1] if len(outs) == 1 else torch.cat([t for _, t in outs], dim=1)

    @torch.inference_mode()
    def forward_batch(self, token_ids: torch.Tensor, positions: torch.Tensor, seqs: list[SeqInput],
                      logits: str = "last", paged=None, slot_mapping: torch.Tensor | None = None,
                      decode_meta: DecodeMeta | None = None) -> torch.Tensor:
        """One forward pass over several sequences, each either PREFILLING (many tokens, empty cache) or
        DECODING (one token on top of its cache).

        token_ids, positions: [N] = all sequences' tokens concatenated, in the order of `seqs`.
        Returns logits [len(seqs), vocab] for each sequence's LAST token ("last"), [N, vocab] ("all"), or
        [R, vocab] for chosen flattened rows when `logits` is an index tensor (M5 verification needs the
        last k+1 rows of a request, and must not pay the LM head for every prompt row).
        """
        for seq in seqs:
            if seq.cache is not None and seq.n_tokens > 1 and seq.cache.length > 0:
                raise NotImplementedError("multi-token step on a non-empty cache (chunked prefill) is not supported")
        assert sum(s.n_tokens for s in seqs) == token_ids.shape[0], "seqs must cover token_ids exactly"
        x = F.embedding(token_ids, self.embed)[None]          # [1, N, hidden]
        cos, sin = self.rope(positions)
        for i in range(self.cfg.n_layers):
            q, k, v = self._pre(i, x, cos, sin, paged, slot_mapping)
            x = self._post(i, x, self._attend(i, q, k, v, seqs, paged, decode_meta))
        for seq in seqs:
            if seq.cache is not None:
                seq.cache.advance(seq.n_tokens)
        return self.lm_head_rows(self.final_norm(x), seqs, logits)

    def lm_head_rows(self, x: torch.Tensor, seqs: list[SeqInput], logits="last") -> torch.Tensor:
        """Final-normed hidden states [1, N, hidden] -> logits for the requested rows (see forward_batch)."""
        if isinstance(logits, torch.Tensor):
            x = x[:, logits, :]
        elif logits == "last":
            # Only score each sequence's next token: a [B, hidden] x [hidden, vocab] matmul, not [N, ...].
            if len(seqs) == 1:
                x = x[:, -1:, :]
            else:
                ends = torch.tensor([s.n_tokens for s in seqs], device=x.device).cumsum(0) - 1
                x = x[:, ends, :]
        return F.linear(x, self.lm_head)[0]

    def forward(self, token_ids: torch.Tensor, positions: torch.Tensor, cache=None,
                logits: str = "last") -> torch.Tensor:
        """ONE sequence (the M1 API): forward_batch with a single SeqInput."""
        return self.forward_batch(token_ids, positions, [SeqInput(token_ids.shape[0], cache)], logits=logits)
