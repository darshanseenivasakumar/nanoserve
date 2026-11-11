"""Drive a RUNNING nanoserve server over HTTP like any OpenAI client would (stdlib only, so `python3` works
without the project venv). Each check prints PASS/FAIL; exit code 1 if any failed.

    python3 .claude/skills/run-nanoserve/client.py --base-url http://localhost:8002
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--base-url", default="http://localhost:8002")
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
args = p.parse_args()
BASE, MODEL = args.base_url, args.model
failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def chat(content, **kw):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": 48, **kw}
    return post("/v1/chat/completions", body)


def text(resp):
    return resp[1]["choices"][0]["message"]["content"]


with urllib.request.urlopen(BASE + "/v1/models") as r:
    served = json.load(r)["data"][0]["id"]
check("GET /v1/models", served == MODEL, served)

# 1. Greedy: an omitted temperature means greedy in nanoserve (OpenAI's default would be 1.0).
q = "Explain what a GPU is in one sentence."
s, b = chat(q)
g1 = b["choices"][0]["message"]["content"] if s == 200 else ""
print(f"\nQ: {q}\nA: {g1}\n")
check("greedy chat returns text", s == 200 and bool(g1.strip()), f"usage={b.get('usage')}")
check("greedy is deterministic", text(chat(q)) == g1)

# 2. Seeded sampling (M13): same seed -> same text, different seed -> different text.
q = "Write a one-line poem about the rain."
s42a = text(chat(q, temperature=0.9, top_p=0.95, seed=42))
s42b = text(chat(q, temperature=0.9, top_p=0.95, seed=42))
s7 = text(chat(q, temperature=0.9, top_p=0.95, seed=7))
print(f"seed 42: {s42a!r}\nseed  7: {s7!r}\n")
check("same seed -> same sampled text", s42a == s42b)
check("different seed -> different text", s42a != s7)

# 3. Streaming (SSE): role chunk, content chunks, finish_reason, [DONE]; measure TTFT and token gaps.
body = {"model": MODEL, "messages": [{"role": "user", "content": "Name three planets."}], "max_tokens": 48,
        "stream": True, "stream_options": {"include_usage": True}}
req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0, first, n, out, done, finish, usage = time.perf_counter(), None, 0, "", False, None, None
with urllib.request.urlopen(req, timeout=120) as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        if line == "data: [DONE]":
            done = True
            continue
        ev = json.loads(line[6:])
        if not ev["choices"]:
            usage = ev.get("usage")
            continue
        ch = ev["choices"][0]
        finish = ch.get("finish_reason") or finish
        delta = ch["delta"].get("content", "")
        if delta:
            first = first or time.perf_counter()
            n += 1
            out += delta
t1 = time.perf_counter()
print(f"stream: {out!r}")
check("stream ends with [DONE] and a finish_reason", done and finish in ("stop", "length"), f"finish={finish}")
check("stream reports usage", bool(usage) and usage["completion_tokens"] <= 48, f"usage={usage}")
if first:
    print(f"       TTFT {1e3 * (first - t0):.0f} ms, {n} chunks, ~{1e3 * (t1 - first) / max(n - 1, 1):.1f} ms/chunk")

# 4. Raw completions endpoint (prompt used as-is, no chat template).
s, b = post("/v1/completions", {"model": MODEL, "prompt": "The capital of France is", "max_tokens": 8})
check("POST /v1/completions", s == 200 and b["usage"]["completion_tokens"] <= 8, repr(b["choices"][0]["text"]))

# 5. Unsupported / invalid parameters are rejected with 400, never silently ignored.
for bad in ({"temperature": -1}, {"top_p": 0}, {"presence_penalty": 0.5}, {"n": 2}):
    s, b = chat("hi", **bad)
    check(f"rejects {bad}", s == 400, b.get("detail", ""))

print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILED: {failures}'}")
sys.exit(1 if failures else 0)
