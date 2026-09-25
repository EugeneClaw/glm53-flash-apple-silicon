#!/usr/bin/env python3
"""E-029: P1 validation ladder — k04 vs control across lengths 8K/16K/24K/32K/56K
(unique prompt per request), plus 256-token decode check at 24K for regression.
Effective rate = prompt_tokens / TTFT (TTFT includes scoring — that's the honest cost)."""
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


def run(prompt, extra, max_tokens=8):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
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
    ctok = 0
    resp = urllib.request.urlopen(req, timeout=1800)
    for line in resp:
        line = line.decode().strip()
        if line.startswith("data: ") and line[6:] != "[DONE]":
            try:
                obj = json.loads(line[6:])
                u = obj.get("usage")
                if u and u.get("completion_tokens"):
                    ctok = u["completion_tokens"]
                ch = obj.get("choices", [])
                if ch and (ch[0].get("delta", {}) or {}).get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
            except Exception:
                pass
    wall = time.perf_counter() - t0
    return ttft, wall, ctok


# char targets for ~K tokens at ~6.8 chars/tok measured
CHARS = {8000: 54_000, 16000: 109_000, 24000: 163_000, 32000: 218_000, 56000: 380_000}

K04 = {"specprefill": True, "specprefill_keep_pct": 0.4}
CTRL = {"specprefill": False}

salt = 2026093000
rows = []
# warm the draft path once (positional warmup) with a 24K k04 throwaway
salt += 1
run(gen(CHARS[24000], salt), K04)

for size in [8000, 16000, 24000, 32000, 56000]:
    for tag, extra in [("control", CTRL), ("k04", K04)]:
        salt += 1
        prompt = gen(CHARS[size], salt)
        ttft, wall, ctok = run(prompt, extra)
        # estimate actual prompt tokens from server? use corpus ratio: size param
        rows.append({
            "size": size, "arm": tag, "ttft_s": round(ttft, 2) if ttft else None,
            "wall_s": round(wall, 2),
        })
        print(f"[{size:>5}] {tag}: ttft={ttft:.2f}s wall={wall:.2f}s", flush=True)

# decode regression check at 24K: 256-token decode, k04 vs control
for tag, extra in [("control", CTRL), ("k04", K04)]:
    salt += 1
    prompt = gen(CHARS[24000], salt)
    ttft, wall, ctok = run(prompt, extra, max_tokens=256)
    decode_tps = ctok / (wall - ttft) if ttft and wall > ttft else None
    rows.append({"size": "24K-decode256", "arm": tag, "ttft_s": round(ttft, 2) if ttft else None,
                 "wall_s": round(wall, 2), "decode_tps": round(decode_tps, 1) if decode_tps else None})
    print(f"[decode256] {tag}: ttft={ttft:.2f}s wall={wall:.2f}s decode={decode_tps:.1f} t/s", flush=True)

with open("artifacts/eval/results/specprefill-ladder.json", "w") as f:
    json.dump(rows, f, indent=2)
print("DONE", flush=True)