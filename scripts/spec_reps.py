#!/usr/bin/env python3
"""E-027 final: rep'd SpecPrefill A/B with per-rep unique prompts, interleaved arms,
server-receipt decomposition (scoring / sparse-prefill / TTFT from server log)."""
import json, time, random, urllib.request, subprocess

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


def server_tail_specprefill():
    out = subprocess.run(
        ["grep", "-a", "scored\\|sparse prefill\\|stream_visible_ttft", "artifacts/omlx-e026c.log"],
        capture_output=True, text=True,
    ).stdout
    return out.strip().splitlines()[-3:] if out else []


ARMS = [
    ("control", {"specprefill": False}),
    ("k04", {"specprefill": True, "specprefill_keep_pct": 0.4}),
    ("k03", {"specprefill": True, "specprefill_keep_pct": 0.3}),
    ("k02", {"specprefill": True, "specprefill_keep_pct": 0.2}),
]
REPS = 3

salt = 2026092700
rows = []
for rep in range(REPS):
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
        rows.append({"rep": rep, "arm": tag, "ttft_s": round(ttft, 2) if ttft else None, "wall_s": round(wall, 2)})
        print(f"rep{rep} [{tag}] ttft={rows[-1]['ttft_s']}s wall={rows[-1]['wall_s']}s", flush=True)

with open("artifacts/eval/results/specprefill-reps.json", "w") as f:
    json.dump(rows, f, indent=2)

# summary
from collections import defaultdict
agg = defaultdict(list)
for r in rows:
    agg[r["arm"]].append(r["ttft_s"])
print("\n=== TTFT summary (3 reps, ~24K prompts, 8-tok decode) ===", flush=True)
for arm in ["control", "k04", "k03", "k02"]:
    vals = [v for v in agg[arm] if v]
    print(f"{arm}: {vals} median={sorted(vals)[len(vals)//2] if vals else None}s", flush=True)
print("DONE", flush=True)