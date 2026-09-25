#!/usr/bin/env python3
"""E-040: CLEAN matched SpecPrefill quality evaluation.
Fixes E-039's flaws: 2048-token budget (no cap-truncation), compact enforced
formats, six diverse realistic task families, deterministic scorers, both arms
on IDENTICAL contexts (same salt per task — content identical, only the
specprefill flag differs; no cache reuse since flags differ... actually same
prompt + different flag = same radix prefix — the control arm runs FIRST and
k04 re-runs with cache? NO: prefix cache would serve k04 the CONTROL's full
prefill. To defeat: different salts per arm BUT same task content structure;
facts identical, filler differs. Quality comparison is content-level, not
token-level, so this is valid.
Tasks: recall / multi-constraint / summarization / tool-call / code-edit / cross-context arithmetic."""
import json, time, random, urllib.request, re

PORT = 8008
MODEL = "glm53-flash-mixed48"

FILLER = ("the quick brown fox jumps over lazy dog program memory kernel attention "
          "mechanism transformer layer weights quantization bandwidth latency throughput "
          "measurement silicon metal gpu cpu unified apple studio cluster spark prefill "
          "decode tokens per second benchmark engineering optimisation profiling evidence "
          "hypothesis experiment system architecture implementation performance analysis results").split()

FACTS = [
    ("Marabel Quinstone keeps the saffron keycard in drawer 42.", ["marabel", "quinstone", "saffron", "42"]),
    ("Project ZEPHYRION-9 went live on the fourth of June.", ["zephyrion", "fourth of june"]),
    ("Old Tomas feeds exactly 17 gulls every dawn.", ["tomas", "17"]),
    ("Copper kettle inventory: 1,742 units in the north shed.", ["1,742", "1742"]),
    ("Dr. Yuki Tananaka patented the amber centrifuge in 2019.", ["tananaka", "amber"]),
    ("The night train to Veldsted departs at 11 minutes past midnight.", ["veldsted"]),
    ("Sergeant Pikeson's callsign is BADGER-FIVE.", ["badger"]),
    ("The greenhouse passphrase is 'verdant otter twilight'.", ["verdant otter twilight"]),
    ("Manifest 88-C lists 306 crates of walnut timber.", ["88-c", "306"]),
    ("Emergency blood supply is in fridge B7.", ["b7"]),
    ("Engineer Rhoda Vail wears a titanium ring on her left thumb.", ["rhoda", "titanium"]),
    ("The waterwheel was replaced in 1907 by the Ashford family.", ["1907", "ashford"]),
    ("Three green flares means 'harbour closed'.", ["three green", "green flares"]),
    ("The vault combination begins 4-4-1.", ["4-4-1", "441"]),
    ("Lamp oil is replenished every 11 days.", ["11 days"]),
    ("The ferry 'Salt Maiden' carries at most 63 passengers.", ["salt maiden", "63"]),
    ("Apprentice Fenwick owes the baker 2 shillings.", ["fenwick", "shillings"]),
    ("The Greyfall Ridge map is in the purple ledger's back cover.", ["greyfall", "purple ledger"]),
    ("Harbour bells ring 9 times when the tide turns at dusk.", ["9 times", "nine times"]),
    ("The hermit of Thornwick Island plays a cracked violin on Sundays.", ["thornwick", "violin"]),
]


def gen_filler(target_chars, salt):
    rng = random.Random(salt)
    words = []
    while sum(len(w) + 1 for w in words) < target_chars:
        n = rng.randint(3, 9)
        words.append(" ".join(rng.choice(FILLER) for _ in range(n)) + ".")
    return " ".join(words)


def build_context(target_chars, salt, payload_fn):
    """Filler with payload items embedded evenly; payload_fn(i) -> str."""
    rng = random.Random(salt ^ 0x5eed)
    parts = []
    chars = 0
    payloads = payload_fn()
    interval = target_chars // (len(payloads) + 1)
    pi = 0
    while chars < target_chars:
        sec = []
        while sum(len(w) + 1 for w in sec) < interval:
            n = rng.randint(3, 9)
            sec.append(" ".join(rng.choice(FILLER) for _ in range(n)) + ".")
        parts.append(" ".join(sec))
        chars += sum(len(p) + 1 for p in sec)
        if pi < len(payloads):
            parts.append(payloads[pi])
            chars += len(payloads[pi]) + 8
            pi += 1
    return "\n".join(parts)


# ---------- task definitions ----------

def task_recall(salt):
    payload = [f"NOTE: {f[0]}" for f in FACTS]
    ctx = build_context(163_000, salt, lambda: payload)
    q = ("For EACH NOTE line, output ONE line exactly: 'N: <key-person-or-object> | <key-number-or-place>'. "
         "20 lines total, nothing else, no preamble. Unknown -> 'N: unknown'.")
    def score(text):
        t = text.lower()
        hits = [i for i, (_, keys) in enumerate(FACTS) if any(k in t for k in keys)]
        return len(hits) / 20.0, f"{len(hits)}/20"
    return ctx, q, score


def task_constraints(salt):
    rng = random.Random(salt)
    topics = ["network latency", "cache coherence", "memory bandwidth", "kernel scheduling",
              "power efficiency", "storage tiers", "compression ratios", "queue depth",
              "clock domains", "thermal envelopes"]
    payloads = [f"NOTE: {rng.choice(topics)} measurements were recorded this quarter." for _ in range(12)]
    ctx = build_context(163_000, salt, lambda: payloads)
    q = ("Reply with EXACTLY 5 bullet lines. Each line: '- ' then at most 12 words. "
         "Line 1 must contain the word 'red', line 3 must contain a number, line 5 must end "
         "with the word 'done'. No other text.")
    def score(text):
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        s = 0.0
        if len(lines) == 5: s += 0.3
        elif 4 <= len(lines) <= 6: s += 0.15
        if len(lines) >= 1 and "red" in lines[0].lower(): s += 0.25
        has_num = len(lines) >= 3 and re.search(r"\d", lines[2]) is not None
        ends_done = len(lines) >= 5 and lines[4].lower().rstrip(".!").endswith("done")
        if has_num: s += 0.25
        if ends_done: s += 0.2
        f1 = int(bool(lines) and "red" in lines[0].lower())
        return s, f"{len(lines)} lines, flags {f1}/{int(has_num)}/{int(ends_done)}"
    return ctx, q, score


def task_summary(salt):
    sections = [
        ("Deployment Topology", "The cluster spans two racks with 40G interconnects."),
        ("Cache Policy", "Write-through caching with 15-minute TTL windows."),
        ("Failure Modes", "Disk saturation cascades into replication lag."),
        ("Remediation Plan", "Add capacity and rebalance shards in Q3."),
        ("Monitoring gaps", "No alerting on queue depth beyond 80 percent."),
    ]
    payloads = [f"SECTION {i+1}: {t}. {b}" for i, (t, b) in enumerate(sections)]
    ctx = build_context(163_000, salt, lambda: payloads)
    q = ("List each SECTION by its exact title, one line each, format 'N. <title>'. "
         "5 lines only, nothing else.")
    def score(text):
        t = text.lower()
        hits = sum(1 for _, (title, _) in enumerate(sections) if title.lower() in t)
        return hits / 5.0, f"{hits}/5 titles"
    return ctx, q, score


def task_toolcall(salt):
    tools = (
        "You have these tools: get_weather(city, date), book_flight(from, to, date, passengers), "
        "search_docs(query, max_results), send_email(to, subject, body), "
        "create_calendar_event(title, date, attendees)."
    )
    payload = [tools,
               "NOTE: the client is in Leeds and wants to fly to Osaka with 2 colleagues on the 14th.",
               "NOTE: they need the forecast for Osaka that day before booking."]
    ctx = build_context(163_000, salt, lambda: payload)
    q = ("Output the sequence of tool calls needed, one JSON object per line, fields "
         "\"tool\" and \"args\". Only the JSON lines, nothing else.")
    def score(text):
        t = text.lower()
        s = 0.0
        if "get_weather" in t and "osaka" in t: s += 0.35
        if "book_flight" in t and "leeds" in t and "osaka" in t: s += 0.35
        if '"passengers": 3' in t.replace("'", '"') or "passengers" in t and "3" in t: s += 0.15
        if t.count('"tool"') >= 2 or t.count("'tool'") >= 2 or t.count("tool") >= 2: s += 0.15
        return min(s, 1.0), f"weather={int('get_weather' in t)} flight={int('book_flight' in t)}"
    return ctx, q, score


def task_codeedit(salt):
    code = (
        "def process_batch(items, threshold):\n"
        "    results = []\n"
        "    for item in items:\n"
        "        if item.score > threshold:\n"
        "            results.append(item.name.upper())\n"
        "    return results"
    )
    payload = [f"CODE FILE batch_utils.py:\n{code}",
               "NOTE: rename the function to filter_strong_items and return names in "
               "lowercase instead of uppercase."]
    ctx = build_context(163_000, salt, lambda: payload)
    q = "Output the complete modified function only, no explanation."
    def score(text):
        t = text.lower()
        s = 0.0
        if "def filter_strong_items" in t: s += 0.4
        if ".lower()" in t and ".upper()" not in t: s += 0.3
        if "for item in items" in t or "for" in t and "items" in t: s += 0.15
        if "score > threshold" in t or "threshold" in t: s += 0.15
        return min(s, 1.0), f"rename={int('filter_strong_items' in t)} lower={int('.lower()' in t)}"
    return ctx, q, score


def task_arith(salt):
    payload = [
        "NOTE: warehouse A holds 1,240 pallets at start.",
        "NOTE: 380 pallets arrive at A on Tuesday.",
        "NOTE: 155 pallets ship out of A on Wednesday.",
        "NOTE: warehouse B holds 970 pallets and receives none.",
        "NOTE: the merger combines A and B inventories.",
    ]
    ctx = build_context(163_000, salt, lambda: payload)
    q = ("Compute the combined total after all events, output exactly one line: "
         "'TOTAL: <number>' then stop.")
    def score(text):
        m = re.search(r"total[:\s]*(\d[\d,]*)", text.lower())
        if not m: return 0.0, "no total"
        val = int(m.group(1).replace(",", ""))
        ok = val == 1240 + 380 - 155 + 970
        return (1.0 if ok else 0.0), f"got {val} (want 2435) {'OK' if ok else 'WRONG'}"
    return ctx, q, score


TASKS = [
    ("recall", task_recall),
    ("constraints", task_constraints),
    ("summary", task_summary),
    ("toolcall", task_toolcall),
    ("codeedit", task_codeedit),
    ("arith", task_arith),
]

ARMS = [("control", {"specprefill": False}), ("k04", {"specprefill": True, "specprefill_keep_pct": 0.4})]


def run(ctx, q, extra):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": ctx + "\n\n" + q}],
            "max_tokens": 2048, "temperature": 0, "stream": True}
    body.update(extra)
    req = urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, chunks, ctok, ptok = None, [], 0, 0
    for line in urllib.request.urlopen(req, timeout=1800):
        line = line.decode().strip()
        if line.startswith("data: ") and line[6:] != "[DONE]":
            try:
                o = json.loads(line[6:])
                u = o.get("usage")
                if u:
                    ctok = u.get("completion_tokens") or ctok
                    ptok = u.get("prompt_tokens") or ptok
                ch = o.get("choices", [])
                if ch and (ch[0].get("delta", {}) or {}).get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks.append(ch[0]["delta"]["content"])
            except Exception:
                pass
    wall = time.perf_counter() - t0
    return ttft, wall, ptok, ctok, "".join(chunks)


results = []
base_salt = 2026095000
for name, builder in TASKS:
    for rep in range(2):
        for arm, extra in ARMS:
            base_salt += 1
            ctx, q, score = builder(base_salt)
            ttft, wall, ptok, ctok, text = run(ctx, q, extra)
            sc, detail = score(text)
            row = {"task": name, "rep": rep, "arm": arm, "ttft_s": round(ttft, 2),
                   "wall_s": round(wall, 2), "ptok": ptok, "ctok": ctok,
                   "score": round(sc, 3), "detail": detail, "text_head": text[:200]}
            results.append(row)
            print(f"[{name} r{rep} {arm}] score={sc:.2f} ({detail}) ttft={ttft:.1f}s ctok={ctok} "
                  f"head={text[:60]!r}", flush=True)

with open("artifacts/eval/results/e040-quality-clean.json", "w") as f:
    json.dump(results, f, indent=2)

# aggregate
from collections import defaultdict
agg = defaultdict(list)
for r in results:
    agg[(r["task"], r["arm"])].append(r["score"])
print("\n=== AGGREGATE (mean score per task/arm) ===", flush=True)
tots = defaultdict(list)
for (task, arm), scores in sorted(agg.items()):
    m = sum(scores) / len(scores)
    tots[arm].append(m)
    print(f"{task:>12} {arm:>8}: {m:.2f}", flush=True)
for arm, ms in tots.items():
    print(f"OVERALL {arm}: {sum(ms)/len(ms):.3f}", flush=True)
print("E040-DONE", flush=True)