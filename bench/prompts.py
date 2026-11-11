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


def shared_prefix_text(tokenizer, prompts: list[dict], n_tokens: int) -> str:
    """M12 workload: a deterministic ~n_tokens text (the "long" prompts' pasted paragraphs, in id order,
    repeated as needed, cut to n_tokens), used as a system prompt shared by every request -- the case
    prefix caching is built for (long system prompts, a shared document, few-shot examples)."""
    longs = [p["prompt"] for p in sorted(prompts, key=lambda p: p["id"]) if p["length"] == "long"]
    text, ids = "", []
    while len(ids) < n_tokens:
        text = (text + "\n\n" if text else "") + "\n\n".join(longs)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return tokenizer.decode(ids[:n_tokens])


def build_requests(tokenizer, prompts: list[dict], num_prompts: int, max_tokens: int, seed: int,
                   max_tokens_range: tuple[int, int] | None = None,
                   shared_prefix: str | None = None) -> list[Request]:
    """Shuffle prompts with `seed`, then take num_prompts of them (cycling if more than the set).

    max_tokens_range=(lo, hi): give each request its own output length, drawn uniformly from [lo, hi] by a
    SEPARATE seeded RNG (so the prompt order is identical to the fixed-length workload). Real traffic has
    varied output lengths, and that is where static batching loses: a batch waits for its longest member.
    """
    order = list(prompts)
    random.Random(seed).shuffle(order)
    len_rng = random.Random(seed + 7919)
    requests = []
    for i in range(num_prompts):
        p = order[i % len(order)]
        messages = [{"role": "user", "content": p["prompt"]}]
        if shared_prefix:                           # M12: the same system prompt in front of every request
            messages.insert(0, {"role": "system", "content": shared_prefix})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        n_out = len_rng.randint(*max_tokens_range) if max_tokens_range else max_tokens
        requests.append(Request(id=i, prompt=text, prompt_tokens=n_tokens, max_tokens=n_out))
    return requests
