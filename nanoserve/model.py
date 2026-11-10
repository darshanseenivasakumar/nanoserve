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
        sd: dict[str, torch.Tensor] = {}
        for f in sorted(folder.glob("*.safetensors")):
            sd.update(load_file(str(f), device=str(device)))
        sd = {k: v.to(dtype) for k, v in sd.items()}
        return cls(cfg, sd)

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

    def _attention(self, i: int, layer: LayerWeights, h: torch.Tensor, cos, sin, cache) -> torch.Tensor:
        B, T, _ = h.shape
        hd = self.cfg.head_dim
        q = F.linear(h, layer.q_w, layer.q_b).view(B, T, -1, hd).transpose(1, 2)   # [B, n_heads, T, hd]
        k = F.linear(h, layer.k_w, layer.k_b).view(B, T, -1, hd).transpose(1, 2)   # [B, n_kv_heads, T, hd]
        v = F.linear(h, layer.v_w, layer.v_b).view(B, T, -1, hd).transpose(1, 2)
        c, s = cos.unsqueeze(1), sin.unsqueeze(1)                                   # broadcast over heads
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        if cache is not None:
            # Store this step's K/V, get back K/V for ALL tokens so far (cached + new).
            k, v = cache.write(i, k, v)
        # is_causal only for multi-token steps: during prefill token t must not see tokens > t.
        # A single decode token may see everything in the cache, so no mask is needed.
        # NOTE: SDPA's is_causal assumes queries and keys start at the same position, so a multi-token
        # step on top of a non-empty cache (chunked prefill) would be WRONG here. Engine forbids it (M1).
        o = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1, scale=hd ** -0.5, enable_gqa=True)
        o = o.transpose(1, 2).contiguous().reshape(B, T, -1).contiguous()
        return F.linear(o, layer.o_w)

    @torch.inference_mode()
    def forward(self, token_ids: torch.Tensor, positions: torch.Tensor, cache=None,
                logits: str = "last") -> torch.Tensor:
        """token_ids, positions: [T] for ONE sequence. Returns logits [T, vocab] ("all") or [1, vocab] ("last").

        With a cache, K/V for these T tokens are appended to it and attention covers all cached tokens.
        """
        if cache is not None and token_ids.shape[0] > 1 and cache.length > 0:
            raise NotImplementedError("multi-token step on a non-empty cache (chunked prefill) is not supported")
        x = F.embedding(token_ids, self.embed)[None]          # [1, T, hidden]
        cos, sin = self.rope(positions)
        eps = self.cfg.rms_norm_eps
        for i, layer in enumerate(self.layers):
            x = x + self._attention(i, layer, rms_norm(x, layer.ln1, eps), cos, sin, cache)
            h = rms_norm(x, layer.ln2, eps)
            x = x + F.linear(F.silu(F.linear(h, layer.gate_w)) * F.linear(h, layer.up_w), layer.down_w)
        if cache is not None:
            cache.advance(token_ids.shape[0])
        x = rms_norm(x, self.norm, eps)
        if logits == "last":
            x = x[:, -1:, :]    # only score the next token: a [1, hidden] x [hidden, vocab] matmul, not [T, ...]
        return F.linear(x, self.lm_head)[0]
