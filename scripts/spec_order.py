#!/usr/bin/env python3
"""E-027b: positional-vs-keep test. Order: k03 FIRST, then k04, k02, control.
If k03 eats the ~60s and k04 is fast -> positional (first-specprefill-after-control).
If k04 still slow -> keep=0.4-specific."""
import json, time, random, urllib.request

PORT = 8008
MODEL = "glm53-flash-mixed48"


def gen(target_chars, salt):
    rng = random.Random(salt)
    words = []
    bank = ("the quick brown fox jumps over lazy dog program memory kernel "
            "attention mechanism transformer layer weights quantization bandwidth "
            "latency throughput measurement silicon metal gpu cpu unified apple "
            "studio cluster spark prefill decode tokens per second benchmark "
            "engineering optimisation profiling evidence hypothesis experiment "
            "system architecture implementation performance analysis results").split()
    while sum(len(w) + 1 for w in words) < target_chars:
        n = rng.randint(3, 9)
        words.append(" ".join(rng.choice(bank) for _ in range(n)) + ".")
    return " ".join(words)


ARMS = [
    ("k03", {"specprefill": True, "specprefill_keep_pct": 0.3}),
    ("k04", {"specprefill": True, "specprefill_keep_pct": 0.4}),
    ("k02", {"specprefill": True, "specprefill_keep_pct": 0.2}),
    ("control", {"specprefill": False}),
]

salt = 2026092800
rows = []
for tag, extra in ARMS:
    salt += 1
    prompt = gen(163_000, salt)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8,
        "temperature": 0,
        "stream": True,
    }
    body.update(extra)
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    resp = urllib.request.urlopen(req, timeout=1800)
    for line in resp:
        line = line.decode().strip()
        if line.startswith("data: ") and line[6:] != "[DONE]":
            try:
                obj = json.loads(line[6:])
                ch = obj.get("choices", [])
                if ch and (ch[0].get("delta", {}) or {}).get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
            except Exception:
                pass
    wall = time.perf_counter() - t0
    rows.append({"arm": tag, "ttft_s": round(ttft, 2) if ttft else None, "wall_s": round(wall, 2)})
    print(f"[{tag}] ttft={rows[-1]['ttft_s']}s wall={rows[-1]['wall_s']}s", flush=True)

with open("artifacts/eval/results/specprefill-order.json", "w") as f:
    json.dump(rows, f, indent=2)
print("DONE", flush=True)