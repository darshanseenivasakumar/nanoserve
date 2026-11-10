"""How many concurrent requests fit with M2's contiguous KV cache (2048-token reservation each)?"""
import sys

import torch

sys.path.insert(0, "/home/darshan/nanoserve")
from nanoserve.kv_cache import ContiguousKVCache  # noqa: E402
from nanoserve.model import Qwen2Model  # noqa: E402

m = Qwen2Model.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
free, total = torch.cuda.mem_get_info()
print(f"after model load: free {free / 2**30:.2f} GiB of {total / 2**30:.2f} GiB")
per = ContiguousKVCache(m.cfg, 2048, device="cuda", dtype=m.dtype).reserved_bytes
print(f"reservation per request: {per / 2**20:.1f} MiB (2048 tokens x {m.cfg.kv_bytes_per_token(m.dtype)} B)")
caches = []
try:
    while True:
        caches.append(ContiguousKVCache(m.cfg, 2048, device="cuda", dtype=m.dtype))
except torch.OutOfMemoryError:
    pass
print(f"caches allocated before OOM: {len(caches)} (theoretical {free // per})")
print("note: the real limit is lower: prefill activations and the LM head also need memory")
