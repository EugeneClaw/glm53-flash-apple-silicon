#!/usr/bin/env python3
"""P1 SpecPrefill A/B v2 (E-027): UNIQUE prompt per arm (defeats prefix cache),
streaming TTFT separation, per-arm receipts. Control vs keep 0.4/0.3/0.2."""
import json, time, random, urllib.request

PORT = 8008
MODEL = "glm53-flash-mixed48"


def gen_corpus(target_chars, salt):
    rng = random.Random(salt)
    words = []
    bank = ("the quick brown fox jumps over lazy dog program memory kernel "
            "attention mechanism transformer layer weights quantization bandwidth "
            "latency throughput measurement silicon metal gpu cpu unified apple "
            "studio cluster spark prefill decode tokens per second benchmark "
            "engineering optimisation profiling evidence hypothesis experiment "
            "system architecture implementation performance analysis results").split()
    while sum(len(w)+1 for w in words) < target_chars:
        n = rng.randint(3, 9)
        words.append(" ".join(rng.choice(bank) for _ in range(n)) + ".")
    return " ".join(words)


# ~24K tokens: bench harness at 96,008 chars gave 14,119 ptok -> ~6.8 chars/tok.
# Target ~24K ptok => ~163,000 chars.
TARGET_CHARS = 163_000

ARMS = [
    ("control-100", {"specprefill": False}, 2026092501),
    ("keep-0.4", {"specprefill": True, "specprefill_keep_pct": 0.4}, 2026092502),
    ("keep-0.3", {"specprefill": True, "specprefill_keep_pct": 0.3}, 2026092503),
    ("keep-0.2", {"specprefill": True, "specprefill_keep_pct": 0.2}, 2026092504),
]


def run_arm(tag, extra, salt, max_tokens=128):
    prompt = gen_corpus(TARGET_CHARS, salt)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    body.update(extra)
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions", data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    chunks = []
    ntok = 0
    resp = urllib.request.urlopen(req, timeout=1800)
    for line in resp:
        line = line.decode().strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        usage = obj.get("usage")
        if usage:
            ntok = usage.get("completion_tokens") or ntok
        ch = obj.get("choices", [])
        if ch:
            d = ch[0].get("delta", {}) or {}
            piece = d.get("content")
            if piece:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks.append(piece)
    wall = time.perf_counter() - t0
    text = "".join(chunks)
    return {
        "arm": tag, "ttft_s": round(ttft, 2) if ttft else None,
        "wall_s": round(wall, 2), "completion_tokens": ntok,
        "text_head": text[:300],
    }


results = []
for tag, extra, salt in ARMS:
    r = run_arm(tag, extra, salt)
    # prompt tokens: read from server log is awkward; compute chars/tok ratio via a tiny probe? no — the usage block in stream may carry prompt tokens on last chunk
    results.append(r)
    print(f"[{tag}] ttft={r['ttft_s']}s wall={r['wall_s']}s ctok={r['completion_tokens']}", flush=True)
    print(f"  head: {r['text_head'][:150]!r}", flush=True)

with open("artifacts/eval/results/specprefill-ab-2.json", "w") as f:
    json.dump(results, f, indent=2)
print("DONE artifacts/eval/results/specprefill-ab-2.json", flush=True)