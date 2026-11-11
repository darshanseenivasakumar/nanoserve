"""Prompt set loading and request building. Uses a fake tokenizer so these run without a download."""
from bench.prompts import DEFAULT_PROMPTS_PATH, build_requests, load_prompt_set, shared_prefix_text


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        return " ".join(f"<{m['role']}> {m['content']} </{m['role']}>" for m in messages) + " <assistant>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def decode(self, ids):
        return " ".join(ids)


def test_shared_prefix_is_identical_in_every_request():
    # M12 workload: one long system prompt shared by every request (where prefix caching pays off).
    prompts = load_prompt_set(DEFAULT_PROMPTS_PATH)
    tok = FakeTokenizer()
    prefix = shared_prefix_text(tok, prompts, 300)
    assert len(tok(prefix)["input_ids"]) == 300
    reqs = build_requests(tok, prompts, 10, 16, seed=0, shared_prefix=prefix)
    ids = [tok(r.prompt)["input_ids"] for r in reqs]
    assert all(i[:302] == ids[0][:302] for i in ids)   # "<system>" + 300 words + "</system>"
    assert len({r.prompt for r in reqs}) == 10 and all(r.prompt_tokens > 302 for r in reqs)
    plain = build_requests(tok, prompts, 10, 16, seed=0)
    assert [r.prompt.split("<user>")[1] for r in reqs] == [r.prompt.split("<user>")[1] for r in plain]


def test_prompt_set_is_well_formed():
    prompts = load_prompt_set(DEFAULT_PROMPTS_PATH)
    assert len(prompts) >= 48
    ids = [p["id"] for p in prompts]
    assert len(ids) == len(set(ids)), "prompt ids must be unique"
    assert {p["length"] for p in prompts} == {"short", "medium", "long"}
    assert all(p["prompt"].strip() for p in prompts)


def test_build_requests_applies_template_and_counts_tokens():
    prompts = [{"id": "a", "length": "short", "prompt": "hello there"}]
    (req,) = build_requests(FakeTokenizer(), prompts, num_prompts=1, max_tokens=32, seed=0)
    assert req.prompt == "<user> hello there </user> <assistant>"
    assert req.prompt_tokens == 5
    assert req.max_tokens == 32
    assert req.id == 0


def test_build_requests_is_deterministic_per_seed():
    prompts = load_prompt_set(DEFAULT_PROMPTS_PATH)
    tok = FakeTokenizer()
    a = [r.prompt for r in build_requests(tok, prompts, 20, 16, seed=0)]
    b = [r.prompt for r in build_requests(tok, prompts, 20, 16, seed=0)]
    c = [r.prompt for r in build_requests(tok, prompts, 20, 16, seed=1)]
    assert a == b
    assert a != c


def test_varied_output_lengths_are_seeded_and_leave_prompt_order_unchanged():
    prompts = load_prompt_set(DEFAULT_PROMPTS_PATH)
    tok = FakeTokenizer()
    fixed = build_requests(tok, prompts, 30, 128, seed=0)
    a = build_requests(tok, prompts, 30, 128, seed=0, max_tokens_range=(16, 256))
    b = build_requests(tok, prompts, 30, 128, seed=0, max_tokens_range=(16, 256))
    assert [r.prompt for r in a] == [r.prompt for r in fixed]       # same prompts, same order
    assert [r.max_tokens for r in a] == [r.max_tokens for r in b]   # reproducible
    assert all(16 <= r.max_tokens <= 256 for r in a)
    assert len({r.max_tokens for r in a}) > 10                      # actually varied


def test_build_requests_cycles_when_asking_for_more_than_the_set():
    prompts = [{"id": str(i), "length": "short", "prompt": f"p{i}"} for i in range(3)]
    reqs = build_requests(FakeTokenizer(), prompts, num_prompts=7, max_tokens=8, seed=0)
    assert [r.id for r in reqs] == list(range(7))
    assert len({r.prompt for r in reqs}) == 3
