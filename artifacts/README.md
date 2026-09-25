# ARTIFACTS — preserved raw receipts

Raw outputs for the decode-era closure verdicts, copied from ephemeral
scratch (a reboot would erase them). Each file is the receipt behind a
ledger entry in `LEDGER.md`.

| File | Backs | What it is |
|---|---|---|
| `e046-decode-census.md` | E-046 | Full decode-path census: per-component kernel/gate truth, dispatch composition, 2,600-3,392 ops/token |
| `e046-census-t1.json` | E-046 | Machine-readable T=1 dispatch census (per-op counts) |
| `e053-acceptance-invalid-h0.json` | E-053 | The invalid acceptance probe (h_prev=0) — kept deliberately as the negative-control receipt |
| `e053b-acceptance-final.json` | E-053b | The valid acceptance probe: pos-1 agreement 0.00, Path A closed |
| `e053b-findings.md` | E-053b | Probe write-up: trunk offline-load recipe, memory numbers, both recipes |
| `e054-d2d3-evidence.md` | E-054 | D2/D3 comparative arithmetic (boundaries decomposed, ceilings, toolchain risk) |
| `e056-decode-distribution.json` | E-056 | Full decode variance distributions: 3 clean launches × n=10 × 2 ctx (the bimodality receipts) |
| `e055-benchmarks-source.md` | E-055 | Audit source for BENCHMARKS.md (numbers cross-checked) |

Prefill-era receipts (E-025…E-045) are reproducible from `scripts/` via the
invocations in `BENCHMARKS.md`; the full private ledger carries their
original paths.
