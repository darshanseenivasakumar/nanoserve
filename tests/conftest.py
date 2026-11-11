"""Shared, expensive fixtures: load each model once per test session.

fp32 comparisons run on the CPU (exact math check, no VRAM pressure); bf16 runs on the GPU (the real
serving dtype). Our fp32 model is built from the HF model's own tensors, so it shares their memory.
"""
import json
import os
from pathlib import Path

import pytest
import torch

MODEL = os.environ.get("NANOSERVE_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROMPTS_PATH = Path(__file__).resolve().parent.parent / "bench" / "prompts.jsonl"


def pick_prompts(ids):
    rows = {json.loads(l)["id"]: json.loads(l) for l in PROMPTS_PATH.read_text().splitlines() if l.strip()}
    return [rows[i]["prompt"] for i in ids]


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="session")
def encode(tokenizer):
    """Chat-template a user message and return its token ids (list[int])."""
    def _encode(text):
        templated = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                  add_generation_prompt=True)
        return tokenizer(templated, add_special_tokens=False)["input_ids"]
    return _encode


@pytest.fixture(scope="session")
def hf_fp32_cpu():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()


@pytest.fixture(scope="session")
def ours_fp32_cpu(hf_fp32_cpu):
    from nanoserve.model import ModelConfig, Qwen2Model
    cfg = ModelConfig.from_hf_config(hf_fp32_cpu.config.to_dict())
    return Qwen2Model(cfg, dict(hf_fp32_cpu.state_dict()))


@pytest.fixture(scope="session")
def hf_bf16_gpu():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()


@pytest.fixture(scope="session")
def ours_bf16_gpu():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from nanoserve.model import Qwen2Model
    return Qwen2Model.from_pretrained(MODEL, device="cuda", dtype=torch.bfloat16)


@pytest.fixture(scope="session")
def target_bf16_gpu():
    """The 1.5B model in bf16 on the GPU, ONE copy for every `-m spec` test (M14: a second module-scoped copy
    next to this one ran the 6 GB GPU out of memory)."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from nanoserve.model import Qwen2Model
    return Qwen2Model.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device="cuda", dtype=torch.bfloat16)
