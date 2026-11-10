"""Prompt set loading and request building. Uses a fake tokenizer so these run without a download."""
from bench.prompts import DEFAULT_PROMPTS_PATH, build_requests, load_prompt_set


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        return f"<user> {messages[0]['content']} </user> <assistant>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


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
