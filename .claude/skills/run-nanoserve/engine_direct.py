"""Drive the engine IN-PROCESS (no HTTP, no child process): the fast path for changes to engine.py,
scheduler.py, kv_*.py, model.py, sampling.py, cuda_graphs.py or kernels/. Builds the engine exactly like the
server does (engine_process.build_engine), batches a few requests, steps until done, prints the text and the
engine's counters.

    uv run python .claude/skills/run-nanoserve/engine_direct.py                    # graphs on (~5 s capture)
    uv run python .claude/skills/run-nanoserve/engine_direct.py --no-graphs        # eager, quicker start
    uv run python .claude/skills/run-nanoserve/engine_direct.py --prefix-caching
"""
import argparse
import time

from transformers import AutoTokenizer

from nanoserve.engine_process import EngineSpec, build_engine
from nanoserve.sampling import SamplingParams

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
p.add_argument("--max-new", type=int, default=32)
p.add_argument("--no-graphs", action="store_true", help="eager steps (no CUDA graphs)")
p.add_argument("--no-fused-ops", action="store_true")
p.add_argument("--prefix-caching", action="store_true")
args = p.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
spec = EngineSpec(model=args.model, max_num_seqs=8, kv_cache_tokens=16384,
                  cuda_graphs=False if args.no_graphs else None,
                  piecewise_graphs=False if args.no_graphs else None,
                  fused_ops=False if args.no_fused_ops else None, prefix_caching=args.prefix_caching)
t0 = time.perf_counter()
eng = build_engine(spec)
capture = eng.capture_graphs() if not args.no_graphs else 0.0
print(f"engine built in {time.perf_counter() - t0:.1f} s (graph capture {capture:.1f} s)")


def encode(text):
    t = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return tok(t, add_special_tokens=False)["input_ids"]


requests = [("Name a color.", None),
            ("What is 2 + 2?", None),
            ("Write a one-line poem about the sea.", SamplingParams(temperature=0.9, top_p=0.95, seed=1))]
ids = {eng.add_request(encode(q), args.max_new, sampling=s): q for q, s in requests}   # batched together
outs = {sid: [] for sid in ids}
t0 = time.perf_counter()
while eng.has_unfinished():
    for o in eng.step():
        outs[o.seq_id].append(o.token)
dt = time.perf_counter() - t0
for sid, q in ids.items():
    print(f"\nQ: {q}\nA: {tok.decode(outs[sid], skip_special_tokens=True)!r}")
st = eng.stats
n_tok = sum(len(v) for v in outs.values())
print(f"\n{n_tok} tokens in {dt:.2f} s; steps {st['steps']} (graph {st['graph_steps']}, piecewise "
      f"{st['piecewise_steps']}); fused_ops {st['fused_ops']}; prefix_caching {st['prefix_caching']}")
