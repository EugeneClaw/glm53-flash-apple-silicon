#!/usr/bin/env python3
"""E-031: HC single-GEMM A/B — prefill ladder (specprefill OFF per request).
Compares against ledger baselines (m48-base, kern2-full). Quality gate: 3 fixed QA pairs."""
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


def run(prompt, max_tokens=8, extra=None):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    if extra:
        body.update(extra)
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    ctok = 0
    ptok = 0
    resp = urllib.request.urlopen(req, timeout=1800)
    for line in resp:
        line = line.decode().strip()
        if line.startswith("data: ") and line[6:] != "[DONE]":
            try:
                obj = json.loads(line[6:])
                u = obj.get("usage")
                if u:
                    ctok = u.get("completion_tokens") or ctok
                    ptok = u.get("prompt_tokens") or ptok
                ch = obj.get("choices", [])
                if ch and (ch[0].get("delta", {}) or {}).get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
            except Exception:
                pass
    wall = time.perf_counter() - t0
    return ttft, wall, ptok, ctok


SP_OFF = {"specprefill": False}

# ladder: 8K/16K/24K/32K/56K x 2 reps, unique prompts
CHARS = {8000: 54_000, 16000: 109_000, 24000: 163_000, 32000: 218_000, 56000: 380_000}
salt = 2026093100
rows = []
for rep in range(2):
    for size in [8000, 16000, 24000, 32000, 56000]:
        salt += 1
        prompt = gen(CHARS[size], salt)
        ttft, wall, ptok, ctok = run(prompt, extra=SP_OFF)
        rows.append({"size": size, "rep": rep, "ttft_s": round(ttft, 2) if ttft else None,
                     "ptok": ptok})
        print(f"rep{rep} [{size:>5}] ttft={ttft:.2f}s ptok={ptok}", flush=True)

# quality gate: 3 fixed QA pairs (non-stream for full text)
QA = [
    ("What is the capital of France? Answer with the city name only.", "Paris"),
    ("What is 17 * 23? Answer with the number only.", "391"),
    ("Name the first three elements of the periodic table.", "hydrogen"),
]
print("\n=== quality gate ===", flush=True)
for q, expect in QA:
    ttft, wall, ptok, ctok = run(q, max_tokens=48, extra=SP_OFF)
    # non-stream: re-request without stream
    body = {"model": MODEL, "messages": [{"role": "user", "content": q}],
            "max_tokens": 48, "temperature": 0, "specprefill": False}
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    raw = json.loads(urllib.request.urlopen(req, timeout=300).read())
    text = raw["choices"][0]["message"]["content"]
    ok = expect.lower() in text.lower()
    print(f"[{'OK' if ok else 'FAIL'}] {q[:40]!r} -> {text[:80]!r}", flush=True)
    rows.append({"qa": q, "expect": expect, "text": text[:200], "ok": ok})

with open("artifacts/eval/results/hc-gemm-ladder.json", "w") as f:
    json.dump(rows, f, indent=2)
print("DONE", flush=True)