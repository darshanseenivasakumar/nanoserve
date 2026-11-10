"""Load the fixed prompt set and turn it into Requests.

We apply the chat template ourselves and send the resulting raw text to every engine. That way all
engines see byte-identical input, and we can check each engine's reported prompt token count against ours.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from bench.backends.base import Request

DEFAULT_PROMPTS_PATH = Path(__file__).with_name("prompts.jsonl")


def load_prompt_set(path: str | Path = DEFAULT_PROMPTS_PATH) -> list[dict]:
    rows = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_requests(tokenizer, prompts: list[dict], num_prompts: int, max_tokens: int, seed: int) -> list[Request]:
    """Shuffle prompts with `seed`, then take num_prompts of them (cycling if more than the set)."""
    order = list(prompts)
    random.Random(seed).shuffle(order)
    requests = []
    for i in range(num_prompts):
        p = order[i % len(order)]
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": p["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        requests.append(Request(id=i, prompt=text, prompt_tokens=n_tokens, max_tokens=max_tokens))
    return requests
