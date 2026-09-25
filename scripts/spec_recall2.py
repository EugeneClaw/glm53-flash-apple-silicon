#!/usr/bin/env python3
"""E-027c: fair recall test. Compact fixed-format answer (no list-length bias),
1024 max tokens, 2 reps per arm (control + k04 only — the quality-viable arm)."""
import json, time, random, urllib.request

PORT = 8008
MODEL = "glm53-flash-mixed48"

FACTS = [
    "Marabel Quinstone keeps the saffron keycard in drawer 42.",
    "Project codename ZEPHYRION-9 went live on the fourth of June.",
    "The lighthouse keeper, Old Tomas, feeds exactly 17 gulls every dawn.",
    "Copper kettle inventory stands at 1,742 units in the north shed.",
    "Dr. Yuki Tananaka patented the amber centrifuge in 2019.",
    "The night train to Veldsted departs at 11 minutes past midnight.",
    "Sergeant Pikeson's radio callsign is BADGER-FIVE.",
    "The greenhouse passphrase is 'verdant otter twilight'.",
    "Cargo manifest 88-C lists 306 crates of walnut timber.",
    "The clinic's emergency blood supply is stored in fridge B7.",
    "Chief engineer Rhoda Vail wears a titanium ring on her left thumb.",
    "The mill's waterwheel was replaced in 1907 by the Ashford family.",
    "Signal fire protocol: three green flares means 'harbour closed'.",
    "The archive vault combination begins with the digits 4-4-1.",
    "Lighthouse lamp oil is replenished every 11 days by the harbormaster.",
    "The ferry 'Salt Maiden' carries at most 63 passengers.",
    "Apprentice Fenwick owes the baker exactly 2 shillings.",
    "The map to Greyfall Ridge is sewn inside the purple ledger's back cover.",
    "Harbour bells ring 9 times when the tide turns at dusk.",
    "The hermit of Thornwick Island plays a cracked violin on Sundays.",
]

FILLER_BANK = ("the quick brown fox jumps over lazy dog program memory kernel "
               "attention mechanism transformer layer weights quantization bandwidth "
               "latency throughput measurement silicon metal gpu cpu unified apple "
               "studio cluster spark prefill decode tokens per second benchmark "
               "engineering optimisation profiling evidence hypothesis experiment "
               "system architecture implementation performance analysis results").split()


def build_fact_prompt(target_chars, salt):
    rng = random.Random(salt)
    parts = []
    fact_interval = target_chars // (len(FACTS) + 1)
    chars = 0
    fi = 0
    while chars < target_chars:
        sec_words = []
        while sum(len(w) + 1 for w in sec_words) < fact_interval:
            n = rng.randint(3, 9)
            sec_words.append(" ".join(rng.choice(FILLER_BANK) for _ in range(n)) + ".")
        parts.append(" ".join(sec_words))
        chars += sum(len(p) + 1 for p in sec_words)
        if fi < len(FACTS):
            parts.append(f"NOTE: {FACTS[fi]}")
            chars += len(FACTS[fi]) + 8
            fi += 1
    parts.append(
        "For EACH of the NOTE lines above, answer on ONE short line in the form "
        "'N: <person/object> | <number/place>'. If you cannot recall a NOTE's "
        "content, write 'N: unknown'. Do all 20, nothing else."
    )
    return "\n".join(parts)


def score_recall(text):
    t = text.lower()
    keys = {
        0: ["marabel", "quinstone", "saffron", "42"],
        1: ["zephyrion", "fourth of june", "june"],
        2: ["tomas", "17", "gulls"],
        3: ["1,742", "1742", "walnut"],
        4: ["yuki", "tananaka", "amber"],
        5: ["veldsted", "11", "midnight"],
        6: ["badger", "pikeson"],
        7: ["verdant otter twilight", "passphrase"],
        8: ["88-c", "306", "walnut"],
        9: ["b7", "blood"],
        10: ["rhoda", "vail", "titanium"],
        11: ["1907", "ashford"],
        12: ["three green", "harbour closed"],
        13: ["4-4-1", "441"],
        14: ["11 days", "oil"],
        15: ["salt maiden", "63"],
        16: ["fenwick", "shillings"],
        17: ["greyfall", "purple ledger"],
        18: ["9 times", "nine times", "tide"],
        19: ["thornwick", "violin"],
    }
    return [i for i in range(20) if any(k in t for k in keys[i])]


ARMS = [
    ("control", {"specprefill": False}),
    ("k04", {"specprefill": True, "specprefill_keep_pct": 0.4}),
]

salt = 2026092900
results = []
for rep in range(2):
    for tag, extra in ARMS:
        salt += 1
        prompt = build_fact_prompt(163_000, salt)
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0,
            "stream": False,
        }
        body.update(extra)
        req = urllib.request.Request(
            f"http://localhost:{PORT}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        raw = json.loads(urllib.request.urlopen(req, timeout=1800).read())
        wall = time.perf_counter() - t0
        text = raw["choices"][0]["message"]["content"]
        hits = score_recall(text)
        results.append({
            "rep": rep, "arm": tag, "wall_s": round(wall, 2),
            "recall": f"{len(hits)}/20", "hits": hits, "text": text,
        })
        print(f"rep{rep} [{tag}] wall={wall:.1f}s recall={len(hits)}/20 hits={hits}", flush=True)

with open("artifacts/eval/results/specprefill-recall-2.json", "w") as f:
    json.dump(results, f, indent=2)
print("DONE", flush=True)