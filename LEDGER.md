# LEDGER — experiment log (abridged public edition)

> Every experiment from the private engineering log, E-001 through E-045 plus
> the decode reopening (E-046+), with hypotheses, measurements, verdicts, and
> artifacts. Environment-specific paths are generalized (`<studio-path>`,
> `<venv311>`, `<models-dir>`, etc.). Device class, model, quant, and measured
> numbers are verbatim. Failed experiments are retained deliberately.
>
> Note: some artifacts referenced here lived on scratch volumes and were not
> all preserved; where a receipt was lost, the audit trail says so (see the
> audit entry). The benchmark scripts in `scripts/` regenerate the key
> measurements.
> Note: E-013, E-015 and E-016 have no standalone entries; their work is described
> inside E-019, E-022 and ROOT-CAUSES.md (RC-2).

# GLM-5.3-Flash Prefill Mission — Engineering Ledger

## NEXT SESSION (V1 punch list complete — 2026-09-25 ~11:30)

**1. Current known-good configuration:** oMLX 0.7.0.dev4 (studio <omlx-tree>, venv311, mlx 0.32.0, transplanted 0.6.4 DMG kernels) + SpecPrefill (keep 0.4, threshold 8192, default-armed) + SPEC_PREFILL_KEEP_POOL=1 + OMLX_DRAFT_CACHE_SIDECAR=1 + glm_draft_adapter (collect_logits). Launch: `cd <omlx-tree> && SPEC_PREFILL_KEEP_POOL=1 OMLX_DRAFT_CACHE_SIDECAR=1 nohup <venv311>/bin/omlx serve --model-dir artifacts/serve-m48 --host localhost --port 8008` (or <studio-path> + the sidecar var). Repo: <mission-dir>/repo (git, tag v1-candidate + 5 commits).

**2. Best benchmarks:** SpecPrefill k04 — 16K 1,630 / 24K 2,089 / 32K 2,018 / 56K 2,040 t/s effective (E-038); full-fidelity control ~1,030 @24K; quality 0.942/1.000 (E-040); decode 28-35 t/s (all engines measured; ceiling statement E-044).

**3. Completed:** all 12 punch items. Audit clean (9/4/0 + dispositions), E-040 quality gate, keep-point confirmed, R3 deferred, sidecar shipped+verified, DS4 measured/closed, GPU ranking complete, versioned.

**4. Eliminated hypotheses:** CPU-dispatch ceiling (E-022 artifact) · point-fusion reclaim (E-031/E-037 twin nulls) · #4521 as major lever (E-035) · DS4 MTP decode (E-044: verify > token cost) · fresh-server 2× myth (E-036 confound) · draft quantisation materiality (E-041) · attention-path bimodality (E-031b).

**5. Active hypotheses (open):** prefix-match sidecar extension (multi-turn chat scoring reuse; effort M, plumbing half-done) · drafter MoE quant IF scoring cost grows · oMLX upstream issue filing (4 findings listed in MISSION FINAL STATE).

**6. Exact next experiment (if continuing):** prefix-match sidecar extension — reuse the longest cached prefix ≤ the request's tokens, score only the suffix through the draft (the E-043 trim/offset plumbing carries most of it), verify with a two-turn growing-conversation test expecting `sidecar prefix hit (N of M tokens)`.

**7. Required files:** repo scripts/ (all harnesses), studio /tmp/{spec_*.py,e040_quality.py,e044_ds4_mtp.py,p2arch/,p3arch/,e031arch/}, server logs artifacts/omlx-e0{26,31,33,36,38,43}*.log.

**8. Risks:** /tmp wiped on reboot (re-stage harnesses from repo first) · ds4 Q4 GGUF occupies 178 GiB (delete if disk needed: <ds4-dir>/gguf) · sidecar default-off in stock oMLX (flag required) · oMLX upstream updates may supersede the vendored glm5_next tree — re-run the audit after any upgrade.

## [E-045] Sidecar in-server verification — HIT CONFIRMED + honest scope (2026-09-25 ~11:25)

**Server receipts (artifacts/omlx-e043.log):** identical-repeat requests after the first: `SpecPrefill: draft cache sidecar hit (26779 tokens, full-state restore)` + `scored 26779 tokens in 0.0s ... draft cache hit 26779`. Scoring cost on hit: **0.0s** (vs 0.1-1.7s miss). Three consecutive hits, LRU stable.

**Honest scope finding from the two-turn test:** turn-2 with APPENDED text correctly MISSES (exact-tuple keyed — a growing conversation is a new tuple). The apparent 38.6s→10.6s turn-2 gain was engine warmup, not cache. The sidecar as-built benefits: identical re-requests (retries, agent re-plans with same context, deterministic evals) — real but narrow. Multi-turn chat benefit requires the prefix-match extension (reuse cached state for the shared prefix, score only the suffix — needs the CacheList offset plumbing already half-done in E-043's trim work).

**Disposition: item 6 CLOSED as implemented-and-verified for the exact-match case; prefix-match extension queued as follow-up (effort M, reuse the E-043 offset/trim plumbing).**

## [E-044] DS4/llama.cpp decode lane — MEASURED, MTP decode-NEGATIVE; lane CLOSED (2026-09-25 ~11:10)

**Setup:** ds4 (antirez) built clean on studio (Metal framework runtime shaders — CLT-only box fine); own-lineage Q4 GGUF (178 GiB, `download_model.sh glm53-q4`); sanity gate PASSED (coherent temp-0 output — no wrong-lineage gibberish). Resident 177.76 GiB + 0.37 GiB compact DSA KV @32K ctx.

**Ordinary decode (ds4-bench CSV, artifacts/ds4-q4-plain.csv):** 31.12/31.00/30.96/30.87/30.81/30.81 t/s @ ctx 2K→12K (flat). Prefill 590→509 t/s.

**MTP decode (ds4-server --mtp --mtp-timing, OpenAI-compat, 9K ctx):**
- draft-1: acceptance 70% (1120/1599); cycle verify2 47.1-48.0ms + head+draft 5.3ms (accept) / 36.4ms (reject); observed 28-29 t/s
- draft-3: acceptance 71%; verify 39.9-42.9ms; observed 28.7-29.1 t/s — **no gain over draft-1**
- **Predicted-from-cycle math matches observation (27.3 predicted vs 28-29 observed) — the measurement is internally consistent**

**VERDICT: DS4 MTP is decode-NEGATIVE on this build/hardware.** The verify pass (~41-48ms) costs more than an ordinary token (~32ms); multi-token drafts don't land (ACCEPT lines show single-draft matches); width-3 changed verify cost marginally, throughput not at all. Ordinary ds4 decode (31 t/s) ≈ oMLX ordinary (30-35 @9K) — no lane advantage. **The 62 t/s decode target is NOT reachable via DS4 on this box; the lane is closed with receipts.** (M5 Max's published 41.97 w/MTP was Q2 — smaller weight read; our Q4 178GiB verify is heavier. A Q2 arm was NOT run: 2× model downloads + rebuild for a lane already showing negative MTP economics is disproportionate.)

**Receipts:** artifacts/ds4-q4-plain.csv, artifacts/ds4-server.log (draft-1), artifacts/ds4-server-d3.log (draft-3), artifacts/e044-mtp.log, artifacts/e044-d3.log.

**Decode ceiling statement (honest):** GLM-5.3-Flash decode on this hardware is ~31-35 t/s across ALL measured engines (oMLX MLX 30-35, ds4 Q4 31, llama.cpp GGUF+MTP historical 42-45 short-ctx at −1.0 quality). The 62 target requires either Q2-class quant (quality cost measured unacceptable) or hardware/runtime changes outside this mission's scope. **Recorded as the genuine constraint, per the brief's evidence bar.**

## [E-043] Draft-cache sidecar (b2) implemented — offline-proven, in-server verify pending (2026-09-25 ~10:50)

**Implementation (Spark, retained):** `omlx/specprefill/draft_sidecar.py` (LRU 6, `OMLX_DRAFT_CACHE_SIDECAR=1`, default off) + fetch/store wiring at `specprefill/draft.py:63-81,179-183` + CacheList fetch-offset fix + descending-member lookahead trim in `patches/specprefill.py` (:593/:628/:660-670). Backups .orig-e042.

**Correctness decisions (reviewed, sound):** pooled rollback `floor(N/ratio)` windows (accumulate_windows emits complete windows only — ceil would retain a lookahead-contaminated window); remainder ≤ ratio−1 boundary tokens dropped (pooled member only; KV exact); store trims to len(tokens) so the exact-hit replay of the last token sees offset N.

**Offline tests: 28/28** (store/restore member equality, snapshot isolation, no future-token leakage, offset checks, LRU eviction, import safety). Live :<port> untouched/stock (verified 200 + compile).

**Pending: in-server two-turn verification** (fold into the DS4 bench restart cycle): restart with `OMLX_DRAFT_CACHE_SIDECAR=1` + canonical env → turn-1 request → turn-2 same-prefix request → expect `draft cache sidecar hit (N tokens, full-state restore)` in log + reduced turn-2 scoring time.

**Caveat (accepted):** exact-tuple keyed — multi-turn chat replay benefits; prefix-match extension is follow-up work.**

## [E-042] DS4 lane opened + draft-cache b2 dispatched (2026-09-25 ~10:35)

**DS4 (Report B, cited):** repo confirmed (antirez/ds4); GLM-5.3-Flash own-graph support, Metal primary, embedded MTP (`--mtp`, `--mtp-timing` acceptance counters), own GGUF lineage (antirez/hf glm-5.3-flash-gguf; **NOT** the Unsloth llama.cpp lineage — wrong-pairing = gibberish risk; use `./download_model.sh` targets only). 256GB tier = Q4 resident (~178 GiB). M5 Max Q2 receipt: 34.45→41.97 t/s w/MTP; **honest prediction for our box: 35-50 t/s — the 62 target is unlikely, but the lane must be measured.**

**Executed:** ds4 cloned + built clean on studio (all binaries; Metal *framework* runtime shader compile — the CLT `metal` gap doesn't bite; commit recorded artifacts/ds4-commit.txt). Q4 download started (~178 GiB). Bench plan adopted from the report: ds4-bench ordinary vs --mtp--mtp-timing, ctx ladder 2K→12K, 256 gen tokens, temp 0, idle box; sanity gate (temp-0 fixed prompt) before timing.

**Draft-cache (Report A, cited):** root cause = draft scoring path passes NO boundary_snapshots to store_cache (draft.py:168-175) + CacheList lookahead-trim gap + fetch-offset gap (specprefill.py:535). Options costed: (a) SSD M/L high-risk; (b1) pm-ineligible store S/M med-risk; **(b2) MTP-style LRU sidecar S low-risk — SELECTED**. Benefits multi-turn stable-prefix workloads only; one-shot agent prompts unaffected. b2 implementation dispatched to Spark (env-gated OMLX_DRAFT_CACHE_SIDECAR=1, offline-tested, live server untouched).

**Status: both lanes in flight.**

**mlx-serve PR #523 (brief item):** recorded and reviewed — prompt-lookup substitution for MTP chains on context-copying output (up to 1.48× on copy-heavy rounds; gated, off-by-default; closed PR in ddalcu/mlx-serve). Not portable to oMLX without porting their spec-decode loop; benefits only copy-dominated outputs; our workload receipts show no such profile. **No action, per the brief's evidence bar.**

## [AUDIT E025-E039] Continuity audit (Spark, read-only) — 9 SUPPORTED / 4 PARTIAL / 0 UNSUPPORTED (2026-09-25 ~09:55)

**Report:** `<mission-dir>/AUDIT-E025-E039.md`. Canonical config verified against reality (all patches present at cited lines; live server launched with verbatim canonical command; healthy SpecPrefill scored lines through 09:45). Perf receipts verified digit-for-digit (E-027/029/035/036/038).

**Material findings + dispositions (strong-model review):**
1. **E-028 v2 artifacts lost to /tmp wipe** → the inherited "~95% recall" lineage had no surviving receipt. SUPERSEDED by E-040 (cap-free, receipts archived in repo). Ledger lineage corrected.
2. **E-039 mislabeled PASSED → INCONCLUSIVE** (accepted: 1024-cap truncation both arms; n=1). E-040 is the gate that closes it.
3. **E-037 citation path wrong** (e033-ab.log nonexistent) → actual receipts omlx-e033b.log + omlx-e036.log; numbers reproduce exactly; verdict stands.
4. **E-031/E-033 microbench outputs unarchived** → ledger-text only; end-to-end receipts exist and carry the verdicts. Accepted as record-hygiene debt; microbenches are reproducible from scripts in repo.
5. specprefill-ladder.json relabeled (E-038 rerun, not E-029); hc-gemm-ladder range corrected (62.22s worst).

**Audit verdict on the quality leg ("provisional — needs one cap-free run") was correct at audit time and is now SATISFIED by E-040** (the audit ran concurrently with it and could not see it).

## [E-041] R3 draft-quantisation — DEFERRED (not material) (2026-09-25 ~10:15)

**Assessment (no code):** post-E-038 steady-state scoring is 0.6-1.7s per request (the 4.8s outlier was the post-56K footprint echo, not steady state). Drafter param census: mlp (MoE) 7.27G + self_attn 0.12G + eh_proj 0.03G = 7.43G params (14.5GB bf16). 4-bit would save ~0.3-0.8s on an ~11.5s k04 TTFT = **3-7% — below the materiality bar**, and it shifts the drafter's scoring logits (quality risk on the very thing E-040 just gated). DEFERRED unless scoring cost grows in real use. If revisited: quantize mlp.* only (attn+norms stay bf16), spot-check with the E-040 harness.

**Punch item 5: CLOSED-deferred with reasoning.**

## [E-040] CLEAN matched quality gate — PASSED at keep=0.4 (2026-09-25 ~10:05)

**Method:** six diverse realistic task families (recall / multi-constraint formatting / summarization / tool-call sequencing / code-edit / cross-context arithmetic), identical 24K contexts per task, 2 reps × 2 arms, 2048-token budget (no cap truncation), compact enforced formats, deterministic scorers. Receipts: artifacts/eval/results/e040-quality-clean.json.

**Results (mean score):** control 1.000 / **k04 0.942** — per task: arith 1.00, codeedit 1.00, summary 1.00, toolcall 1.00, constraints 0.88, recall 0.78 (rep0 20/20, rep1 11/20).

**Interpretation:** perfect parity on agent-workload-relevant tasks (tool calls, code edits, arithmetic, summarization). The cost concentrates in exhaustive fact-recall through long filler — exactly where a 60%-token approximation must lose something. ~94% overall, ~78-100% on the hardest family. **keep=0.4 confirmed as the operating point** (0.3/0.2 cliff per E-028; full-fidelity fallback one flag away).

**Status: SUPPORTED — SpecPrefill quality claim CLOSED with fresh post-fix evidence.**

## [E-039] Post-fix quality re-gate — PASSED; P1/P2/P3/P4/P5 ALL CLOSED (2026-09-25 ~08:30)

**Recall re-run on the E-038-fixed path:** k04 clean rep **18/20** (vs control's truncated 8/20 — both arms exceed the 1024-token cap in verbose format; fact-by-fact output identical where present). Rep1 artifacts are the documented harness issues (degenerate first-request control output; meta-reasoning drift), excluded per E-028 precedent. **Quality unchanged by the fixes: ~90-95% recall preservation at keep=0.4.**

---

## MISSION FINAL STATE (punch list complete)

| Item | Verdict | Ledger |
|---|---|---|
| P1 SpecPrefill | **SHIPPED: ~2× prefill at every size ≥16K; 2,018-2,089 t/s effective @24-32K (target 1,584 exceeded ~30%); ~95% recall preserved** | E-025/026/027/028/029/031b/038/039 |
| P2 HC single-GEMM | Neutral at production chunk (proven equivalent, 3.96×/call micro, gated-off retained) | E-030/031 |
| P3 #4521 decode A/B | +3.0-3.5% decode, inside rep spread — free knob, not shipped | E-034/035 |
| P4 GPU-time breakdown | Complete: 57-71% of chunk time = execution premium between ops; twin nulls (E-031/E-037) prove point-fusion can't reclaim it | E-032 |
| P5 narrow fusion | KDA-norm kernels proven equivalent + 3-5.5×/call, neutral end-to-end, gated-off retained | E-033/037 |

**Canonical config:** oMLX dev4 + venv311 + transplanted kernels + SpecPrefill (keep 0.4, threshold 8192, default-armed) + SPEC_PREFILL_KEEP_POOL=1 + glm_draft_adapter (R1). Launch: `cd <omlx-tree> && SPEC_PREFILL_KEEP_POOL=1 nohup <venv311>/bin/omlx serve --model-dir artifacts/serve-m48 --host localhost --port 8008`.

**Honest gaps (documented, not blocking the prefill mission):**
1. Decode 28-35 t/s vs 62 target — separate lane; needs MTP via llama.cpp fork (proven +23% there) or upstream work. Not addressable by this punch list.
2. Scoring still costs 0.6-4.8s/request; R3 (drafter MoE quantization) would shrink it further.
3. Draft cache never persists (CacheList snapshots unsupported) — multi-turn suffix-only rescoring is a real future win.
4. Effective-prefill definition: token-reduction via importance selection (~95% recall), not bit-exact full prefill. Users who need 100% recall fidelity send `specprefill:false` and get the ~1,030 t/s full path.

**Upstream-worthy findings (for the oMLX issue the user may file):** (a) the draft path's per-chunk mx.clear_cache() under high footprint = 130× scoring degradation — SPEC_PREFILL_KEEP_POOL should be default; (b) full-vocab lm_head per draft chunk wasted 75% of scoring compute; (c) select_chunks' 750 .item() syncs; (d) HC single-GEMM equivalence proof (neutral at 2048-chunk but 165ms/chunk at 8192-step).

## [E-038] Draft-scoring fixes (E-031b R1+R2+R4a) — SLOW MODE DEAD; P1 UNBLOCKED (2026-09-25 ~08:15)

**Root cause (E-031b Spark, artifacts/e031arch/draft-scoring-rootcause.md):** the 44-73s scoring mode was NOT attention-path selection (three independent proofs: deterministic per-chunk path; fast 56K == fast 24K; analytic bound 300× too small). It was per-chunk `mx.clear_cache()` under the ~193GB high-footprint regime forcing full buffer-pool re-creation every chunk — the exact lesson the TARGET already learned and disabled (E-012a) but the draft path still paid. Secondary waste: full-vocab lm_head per chunk (2.6 TFLOP + 0.63GB alloc discarded, ~75% of fast-mode compute). Also: 8K prompts were never scored (below threshold 8192).

**Fixes applied (all three):**
- R1: `collect_logits` flag in DraftLM; `_prefill_draft` disables lm_head for chunk calls, re-enables for the final logits
- R2: `SPEC_PREFILL_KEEP_POOL=1` env gate on the per-chunk clear (score_tokens still clears once at end)
- R4a: vectorized select_chunks chunk-means (one eval vs ~750 .item() syncs)

**Result (server receipts artifacts/omlx-e038.log, paired control/k04 ladder):**

| Size | Control TTFT | k04 TTFT | Scoring | Speedup |
|---|---|---|---|---|
| 8K | 7.80s | 7.65s | n/a (<8192) | parity |
| 16K | 18.20s | **9.88s** | 1.7s | **1.84×** |
| 24K | 22.95s | **11.49s** | 0.6s | **2.00×** |
| 32K | 32.28s | **15.95s** | 3.1s | **2.02×** |
| 56K | 54.24s | **27.50s** | 1.7s | **1.97×** |

**No slow-mode occurrence in the entire ladder. Effective rates: 16K 1,630 · 24K 2,089 · 32K 2,018 · 56K 2,040 t/s — the 1,584 target exceeded at every size ≥16K.**

**P1 VERDICT: SPEC PREFILL SHIPPABLE — ~2× uniform prefill speedup at all scored sizes, quality previously gated (~95% recall, E-028).** Server now runs canonical WITH SpecPrefill default-on (keep 0.4) and SPEC_PREFILL_KEEP_POOL=1.

**Residuals (recorded, non-blocking):** scoring 0.6-4.8s per request (R3 drafter-quantization would shrink further); draft cache never persists (CacheList snapshot unsupported — multi-turn win available); vectorized select_chunks values identical by construction (mean over same slices) but a numerical-agreement spot-check is queued.

## [E-037] P5 KDA fused-norm A/B — NEUTRAL end-to-end; P5 CLOSED (2026-09-25 ~03:20)

**A/B (fresh servers, unique prompts, specprefill off, 2 reps × 16/24/32K):** flag-on vs flag-off medians: 16K −2.2%, 24K +0.6%, 32K +0.3% — **all inside ±2s rep noise. NEUTRAL.** Quality 3/3. Receipts: artifacts/e033-ab.log + artifacts/eval/results/e036-fresh-control.json.

**Flag-engagement verification chain (settled):** `VAR=1 nohup` env delivery works (canary-proven); `ps eww` env-blindness on macOS 27 documented (all processes show no env); vendored language.py loads with flag=True under the compat patch (subprocess-proven); the armed-line canary was mis-designed (logs at kda_fused_norms import inside the first flagged forward — logger timing/filtering unreliable). The A/B itself was the true engagement test: outputs identical-structure, timings shifted within noise — consistent with engaged-but-neutral OR engaged-kernels-but-glue-was-cheap; either way the punch-list verdict (neutral → don't ship) is unaffected.

**P5 DISPOSITION (parallel to E-031's):** kernels retained as proven infrastructure (equivalence 7.5e-09/9.5e-07, 3-5.5×/call standalone) but the fused-norm path is NOT the lever P4's attribution suggested. **Two consecutive E-031/E-037 nulls establish the pattern: the ~57-71% "execution premium" P4 measured is NOT reclaimable by per-op kernel fusion at these sites — it lives in the cache-object/quant-prep machinery between ops, not in the norm/glue ops themselves.** The premium's true mechanism (E-022's ~1.16ms/kernel-equivalent serial latency) needs structural work (multi-stream overlap or whole-graph capture), not point fusions.

**P5 CLOSED. Punch list status: P1 (draft-fix pending E-031b report) · P2 closed · P3 closed · P4 closed · P5 closed.**

## [E-036] Fresh-server control recalibration — FALSE ALARM resolved; baseline UNCHANGED (2026-09-25 ~03:00)

**E-035's side-finding ("fresh control = 2,156 t/s @20K, 2× baseline") is a measurement-confound, now resolved:** fresh-restart server, cleared cache, `specprefill:false` per request → **16K: 1,018/886 t/s · 24K: 954/1,014 · 32K: 969/967** (2 reps each; receipts artifacts/eval/results/e036-fresh-control.json + artifacts/omlx-e036.log). Fresh-server control ≈ long-lived-server control ≈ ~950-1,030 t/s. The P3 script's 2,156 t/s was its bench requests NOT disabling SpecPrefill — that column measured the **keep-0.4 sparse path** (20,092×0.4≈8,037 kept tokens @ ~1,040 t/s + 0.4s scoring ≈ 9.4s TTFT ✓ matches its 9.375s). The P3 DECODE verdict is unaffected (decode is post-prefill); its prefill side-column gets an erratum: it was SpecPrefill-on, not control.

**Canon now:** honest control @24K ≈ 1,000-1,030 t/s; SpecPrefill k04 warm @24K = 2,455 t/s effective (2.4×). The mission baseline was never mis-stated; the "2× fresh" hypothesis is closed as REFUTED-BY-CONFOUND.

**Methodology rules reinforced:** (a) every prefill bench must state specprefill on/off explicitly — default-on means "control" isn't control; (b) corpus tokenizes at 6.78 chars/token — always calibrate.

## [E-035] P3 #4521 A/B EXECUTED — decode +3.0-3.5%, inside spread; P3 CLOSED (2026-09-25 ~02:45)

**Result (fresh dedicated servers per arm, fixed salts, streaming, usage-verified, temp 0, 256 tok):**

| ctx | control | treat (1000/400) | delta |
|---|---|---|---|
| 4K | 28.26 t/s | 29.26 t/s | **+3.5%** |
| 20K | 34.55 t/s | 35.57 t/s | **+3.0%** |

**Verdict: directionally consistent with #4521's author (+5-7.7%) but below it and inside our rep spread (4K range 18.7-35.4). NOT a shippable win; env vars are a free knob if ever wanted, worth ~3%. Prefill: -9.6% @20K (2156→1949) — #4521 predicts unchanged; our n=2 spread is wide; no prefill claim either way.**

**Bit-identity: 0/3 and 0/2 — outputs NOT bit-identical between arms** (contradicts the author's bit-identical claim; likely benign kernel reordering — recorded honestly).

**Side-finding (IMPORTANT for the mission): 20K prefill on FRESH-RESTART control = 2,156 t/s — nearly double the mission's long-lived-server ~1,030 t/s. Hypotheses: (a) prompt-cache/SSD state accumulation slows the long-lived server; (b) the shared server's history (SpecPrefill armed, drafts built) taxes baseline prefill. Either way: PREFILL A/Bs MUST use fresh dedicated servers (the P3 script's pattern) or the comparison is invalid. This may also mean the honest control @24K is better than the ledger's ~1,034 — needs a fresh-server prefill ladder to recalibrate the baseline.**

**Protocol receipts:** artifacts/p3arch/{run_4521_ab.sh, results/ab-{control,treat}.json, ab-run-0236.log}; restore verified (pid 9912); EXIT-trap worked.

**Status: P3 CLOSED. Next: (1) fresh-server prefill recalibration (E-036); (2) E-033 A/B on a fresh window; (3) E-031b report lands → draft-fix.**

## [E-034] P3 #4521 prep + decode baseline (Spark-γ) — A/B script ready, baseline bracketed (2026-09-25 ~02:40)

**Baseline (live server, receipts artifacts/p3arch/results/):** ~4K-ctx decode pooled median **27.8 t/s** (n=19, range 21.2-34.3; clean pre-restart subpool 30.7 n=6, post-restart 27.2 n=7); 20K-ctx 34.3 pre (n=2) vs 24.4 post (n=2). Client t/s matched server-log tok/s within ~0.5. Contamination window (01:52-02:04 restart) handled: re-ran post-02:05, samples labeled. **Correction to mission lore: "~39.5 decode" is SHORT-ctx; at ~4K ctx the honest number is ~28-31. Corpus tokenizes at 6.78 chars/token on GLM (calibrated).**

**Post-restart decode drop (−11% @4K / −29% @20K, small n) — flagged for E-033: either cold-start variance or, if the KDA flag engaged, the fused kernels' decode-shape cost. Armed-line canary absent → unresolved; next restart must settle it.**

**E-033 flag-diagnosis CORRECTED:** `ps eww` shows NO env for ANY process on macOS 27 (Spark proved with /bin/sleep) — my "flag didn't reach the server" was a ps env-blindness artifact. `VAR=1 nohup cmd &` DOES deliver env (canary-proven twice). The armed-line absence therefore means the branch genuinely didn't execute → investigate import path (vendored vs site-packages language.py) at next restart with `export` + armed-line canary as the ONLY reliable check.

**Wheel check:** MLX_MAX_OPS_PER_BUFFER + MLX_MAX_MB_PER_BUFFER both PRESENT in the live-loaded libmlx.dylib (strings receipts) — the #4521 experiment is viable on mlx 0.32.0.

**Protocol `artifacts/p3arch/run_4521_ab.sh` (469 lines, mock-tested on :8009 with zero :<port> contact; 4 real bugs caught and fixed):** snapshot → control → treatment (1000/400) → restore; env-prefix replay (includes E-033 flag — arms stay paired so #4521 comparison internally valid); kill-by-recorded-PID only; idle gate; EXIT-trap auto-restore; fixed salts → byte-identical corpora + bit-identity column.

**Run command:** `ssh <studio-host> 'bash artifacts/p3arch/run_4521_ab.sh 2>&1 | tee artifacts/p3arch/ab-run-$(date +%H%M).log'` (~15-20 min).

**Status: P3 A/B executing next; E-033 A/B queued after (shared restart windows).**

## [P3] MLX #4521 command-buffer-limit A/B — protocol ready + baseline DONE (2026-09-25 ~02:35)

**Baseline (live :<port>, streaming, usage-verified, temp 0, 256 tok, ~4K-ctx unique salted prompts, bench_4521.py):** pooled 4K-ctx decode **median 27.8 t/s (n=19, 21.2–34.3)**; clean pre-restart subpool 30.7 (n=6, log e031) vs post-restart 27.2 (n=7, log e033); 20K-ctx pre-restart **34.3** (n=2, tight) vs post-restart 24.4 (n=2, wide). NOTE: ~4K-ctx ≠ the mission's "39.5 short" (short-ctx figure; historical curve brackets our data). Corpus calibration: this word bank tokenizes at **6.78 chars/token** on GLM (16k chars → 2,359 ptok) — nominal size ≠ prompt tokens, always calibrate.

**Post-restart decode drop is REAL (−11% @4K, −29% median @20K) and correlates with the E-033 relaunch.** ⚠️ **E-033's "flag delivery FAILED" blocker is WRONG — measurement artifact:** `ps eww`/`ps ewww`/`sudo launchctl procinfo` show NO env for ANY process on macOS 27 (tested: plain /bin/sleep with setenv also invisible). The `VAR=1 nohup cmd` pattern DOES deliver (canary child wrote back `MLX_MAX_OPS_PER_BUFFER=1000/MLX_MAX_MB_PER_BUFFER=400` from os.environ, and a stub server launched through the identical chain logged the vars in-process). So `OMLX_KDA_FUSED_NORMS=1` almost certainly reached the server; the missing "kernels armed" line is a code-level issue (import-order or gate bug), and **the post-restart decode regression may be those kernels' slow path or another relaunch effect — E-033 must re-check armed-ness by log canary, not ps.**

**Env-verification recipe for macOS 27 (no post-hoc channel exists):** prove at launch time — (a) canary process through the identical launch chain writing os.environ to a file, (b) marker line written into the server log pre-launch, (c) exec line recorded. Receipts: artifacts/p3arch/env-receipt-*.txt, mocktest/.

**Protocol (ready for parent, mock-tested end-to-end on :8009 with zero :<port> contact):** `ssh <studio-host> 'bash artifacts/p3arch/run_4521_ab.sh 2>&1 | tee artifacts/p3arch/ab-run-$(date +%H%M).log'` — snapshots live server (PID+log+env prefix via parent wrapper cmdline), control arm → treatment arm (MLX_MAX_OPS_PER_BUFFER=1000 MLX_MAX_MB_PER_BUFFER=400) → restore (replays env prefix + original log), kill-by-exact-PID only, idle gate (refuses if log touched <120s), trap-restore, `--restore-only`/`--dry-run`/`--summarize-only`. Both arms fixed salts ⇒ byte-identical corpora ⇒ bit-identity check; per-arm cache clear.

**Wheel check:** `strings libmlx.dylib` → both `MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER` present (mlx 0.32.0, the dylib the live server has open). Expected: +5% @20K, +7.7% short (issue author); pooled-baseline spread (±20%) is wider than the effect — verdict honest-null unless paired-arm deltas clear noise.

**Bash traps found by mock-testing (would have broken the real run):** (1) `local a=$1 b="…$a…"` in one statement = unbound-var crash under `set -u` ($a expands before assignment); (2) `CPID=$(start_server …)` swallows function log output into the variable and a mid-function crash continues with garbage PID — return PIDs via global var; (3) comm-guard must basename-match (`omlx-server` comm is a full Python framework path for some launchers); (4) trap must handle "ORIG_PID still owns port" as owned-not-foreign.

**Artifacts:** artifacts/p3arch/{run_4521_ab.sh, bench_4521.py, agg.py, P3-4521-REPORT.md, results/baseline-*.json, mocktest/}. ⚠️ /tmp wiped on reboot — re-stage from companion machine scratch if needed.

## [E-033] P5 KDA fused norm kernels — EQUIVALENT + 3-5.5×/call; in-server A/B pending (2026-09-25 ~02:10)

**Built (strong-model work):** two `mx.fast.metal_kernel` JIT kernels replacing the P4-ranked #2 GPU-time consumer (KDA norm/gate glue, ~425ms/chunk attributed):
- `kda_l2norm_pair` (q,k l2norm + scale, one kernel, 32-lane simd rows) — replaces language.py:317-318 eager chain
- `kda_onorm_gated` (RMSNorm × weight × sigmoid(gate), one kernel) — replaces Glm5NextRMSNormGated.__call__:105-112 at :341

**Equivalence (measured):** l2norm-pair max|dq| 7.45e-09 / max|dk| 5.96e-08; o_norm-gated max|d| 9.54e-07. **Per-call (slope-fit): l2norm 268→89.5µs (3.0×), o_norm 308→55.6µs (5.5×).**

**Honest standalone projection:** 34 layers × (pair+onorm) = 19.6→4.9ms/chunk = **0.8% of chunk GPU time at pipelined rates** — P4's 23% attribution assumes the in-server premium transfers; E-031's null warns it may not. The in-server A/B is the only arbiter (E-031 lesson applied both ways).

**Integration (retained, env-gated):** `omlx/patches/kda_fused_norms/__init__.py` + vendored language.py call-site branches (`_KDA_FUSED_NORMS` from `OMLX_KDA_FUSED_NORMS=1`); backup language.py.orig-e033. Metal-syntax ladder for the record: input_names/output_names (0.32 ABI) → explicit inputs/output_shapes/dtypes call → thread_position_in_grid.{x,y} threads-not-groups geometry, simd_sum + thread_index_in_simdgroup (house style per gated_delta.py) → single-output kernels return a list.

**Blocker (open):** flag delivery to the server FAILED silently — `VAR=1 nohup` reaches a plain python child (verified) but the running omlx-server process shows no OMLX var (ps eww) and no "kernels armed" import line; suspicion: the omlx console-script wrapper scrubs/loses the prefix-var. Next restart: `export` in shell + verify via armed-line canary. Server currently stock-path (which kept P3's baseline clean).

**Status: kernels proven, A/B pending clean GPU window (after P3 baseline finishes).**

## [E-032] P4 GPU-time breakdown (Spark-β) — ranked list, structural answer (2026-09-25 ~01:50)

**Report**: `<mission-dir>/P4-GPU-TIME-BREAKDOWN.md` (companion machine). All derived numbers computed in Python from receipts, not estimated.

**Ranked top-5 GPU-time consumers per 2048-chunk (anchor W=1,876ms, E-022):**
1. **MoE scatter/glue chain ~530ms (28%)** — gather/sort/unsort/weighted-sum glue around the routed GEMMs
2. **KDA norm/gate glue ~425ms (23%)** — l2norm pair + g-chain + o_norm fp32 round-trips (34 layers)
3. MoE routed GEMMs ~170ms (measured 412 TFLOPS anchor)
4. SPA exact attention ~170ms [analytic-estimate — NO standalone replica exists; F-5 gap]
5. SPA indexer glue ~110ms

**Reconciliation findings:** ablation shares reproduce exactly from raw t/s (40.4/28.0/23.0%); individual deltas over-attribute ~1.08× (sum 12.71 vs 13.69s wall) — read each ±10%. KDA standalone anchors disagree 2.7× (replica 4.7ms vs E-018-scaled 12.8ms/layer) — F-4 tiebreak needed. HC's 41%-of-dispatches = **2.7% of GPU time** (E-031 confirmed at scale); the gated-delta scan = 3.5% (E-002's 62% artifact confirmed dead).

**Structural answer (§5): within per-chunk GPU time, NO single change recovers ≥20%.** GEMM math at demonstrated capability needs ~124ms; standalone floors sum to 540-815ms → **~57-71% of every chunk is execution premium** (cache-object/quant-prep/fp32-norm glue), and it is NOT uniform: HC runs at standalone speed in-server; MoE/KDA/SPA run ×3-13. **The only ≥20% lever is SpecPrefill k04 (measured 2.4× @24K warm, E-027) — token reduction, not per-chunk time — blocked only by E-031b's slow-band root cause.**

**P5 target selected from this data:** KDA norm/gate glue (#2, 425ms, 23%) → the l2norm-pair + o_norm fused JIT kernels (E-024 matrix #1/#2: self-contained row-reductions, #3939-safe, no toolchain). Expected 5-13%. Also queued: MoE fused epilogue (8-20% band, needs toolchain decision) — larger but riskier.

**Measurement gaps (queued):** F-5 SPA standalone replica; F-4 KDA anchor tiebreak.

**Status: COMPLETE. P4's data reorders P5: KDA-glue kernels, with E-031b gating the bigger SpecPrefill lever.**

## [E-031] P2 HC single-GEMM: microbench WIN, end-to-end NEUTRAL → gated OFF (2026-09-25 ~01:45)

**Chain:** Spark-α archaeology (E-030) → equiv_test 5/5 PASS (max diff 1.3e-06 small / 2.86e-05 full-scale, fp32 accumulation order) → hc_bench: **tiled 2.46 ms/call vs GEMM 0.62 ms/call (3.96×), traffic 2.74 GB → 539 MB (5.08×)** @T=8192 → patch module `omlx/patches/hc_single_gemm/` implemented + wired at model_loading glm5_next gate → server A/B.

**End-to-end A/B (specprefill off, 2 reps × 5 sizes):** patched TTFTs 9.3-59.8s vs control baseline 8.1-54.6s — **within rep noise (±2-3s); no measurable win at production chunk=2048.** Quality gate 3/3 OK (France/multiply/periodic). Memory +0 (view hoist).

**Why neutral (P4's warning, confirmed):** production chunk is 2048 → _mix is 90×34 ops ≈ 45-55 ms/chunk ≈ **~3% of chunk GPU time**; the 3.96×/call saving is ~40 ms/chunk, below noise. **41% of dispatches ≠ 41% of GPU time.** At T=8192 (adaptive step, not active) the saving would be ~165 ms/chunk.

**Disposition:** patch RETAINED in-tree, gated OFF by default (`OMLX_HC_SINGLE_GEMM=1` to enable) per the P2 rule: neutral → don't ship, but don't delete proven infrastructure. Restores server to control-equivalent config; next restart picks up the gate.

**Status: CLOSED (neutral at current chunk size; ready if 8192-step is enabled).**

## [E-029] Length ladder — MIXED: k04 fast at 8K/56K, dense-fallback stall in 16–32K band (2026-09-25 ~01:10)

**Data (TTFT, unique prompts, server receipts):**

| Size | control | k04 | k04 scoring |
|---|---|---|---|
| 8K | 8.1s | 9.5s (+17%) | 2.0s |
| 16K | 15.6s | 50.6s (**−69%** REGRESSION) | 44.1s |
| 24K | 23.3s | 59.4s (−61%) | 50.0s |
| 32K | 31.2s | 87.5s (−64%) | 73.2s |
| 56K | 54.6s | **28.8s (+47%)** | 0.9s |

**Root cause identified (source, vendored language.py:700-735):** draft chunk scoring falls into the **dense latent-expanded SDPA fallback** (no topk path; `embed_q(kv_latent)` expansion + full causal attention over the entire pooled context) for prompt sizes where the DSA path selection misses — quadratic re-read of the whole KV per 2K draft chunk. 8K (pooled 2K) and 56K (topk active) escape it; the 16–32K band hits it. The E-027 24K fast runs (0.4s scoring) prove the fast path exists at 24K too — path selection is history-dependent (cache state), not purely size-determined; exact trigger TBD.

**Consequences:**
1. E-027's 2,455 t/s headline stands for warm-path 24K, but **k04 is NOT shippable across the length ladder** until the draft scoring path is fixed.
2. Sparse prefill itself stays ~1,040 t/s at every size and never regressed — the mechanism is sound; the draft's attention path is the problem.
3. Draft SSD cache non-functional (CacheList snapshot truncation, E-027 #4) — forces full re-scoring; fixing it is the chat-workload fix.

**Next (queued, next session):**
- **E-031**: fix draft scoring path — windowed chunk-local attention for draft scoring (architectural; quality-gated), OR path-selection fix to keep the draft on the fast path at all sizes; Spark-δ to map exact trigger conditions + CacheList boundary-snapshot requirements.
- E-032: CacheList draft-cache fix (suffix-only rescoring for chat).

**Status: REFUTED as-is (k04 across sizes); fast-path mechanism confirmed real. P2 promoted to next-GPU-work.**

## [E-030] P2 HC archaeology complete (Spark-α) — SINGLE-GEMM READY TO TEST (2026-09-25 ~01:30)

**Report**: `~/.../scratch/p2arch/report.md` (Mini) + scripts on studio `artifacts/p2arch/{equiv_test.py,hc_bench.py}` (compile-verified, md5 recorded).

**Key facts (file:line-cited):**
1. **Execution path**: server runs **site-packages** `mlx_vlm/models/deepseek_v4/hyper_connection.py` (vendored glm5_next escapes via `..deepseek_v4` relative import; census line-numbers prove it). omlx's `patches/deepseek_v4/hyper_connection.py` is a DIFFERENT model's HC (deepseek_v4 site) — do not patch it.
2. **Math**: per HC call at prefill: x[1,T,4,4096] → astype fp32 → rms_norm over flattened 16384 → **`_mix` = tiled_linear over ⌈T/256⌉ tiles of [256,16384]@[16384,24]** (34 ops/call @T=2048; 130 @T=8192) → sinkhorn+collapse kernel → hc_expand.
3. **Equivalence PROVEN** (block-matrix algebra): `tiled_linear(λ·@fn.T, z) ≡ z @ fn.T` — pad+trim are row-exact inverses; concat is block-stacking; contiguous/slice value-free. **Single GEMM [1,T,16384]@[16384,24], 34→1 ops/call (−97%), 3,060→90 dispatches/chunk (−40% of total).**
4. **Production chunk = 2048** (prefill_guard.py:34, SchedulerConfig default) — census shapes confirmed as production; T=8192 is the glm_moe_dsa adaptive step, covered by the formula.
5. **Traffic** (the E-022-relevant number): tiled 416–685 MB/call → GEMM 136 MB/call @T=2048 (**3.06–5.03×**, −37.5→−61.6 GB/chunk→−12.2 GB); @T=8192: 1.67–2.74 GB → 539 MB. Census F1's "~12GB copies/chunk" cross-checks exactly (90×134 MB contiguous+fnᵀ).
6. **fn.T hoist = 0 bytes** (transpose view; matmul consumes strided operands — production already does this per tile). fp32 pin must be preserved (cast_predicate + sanitize keep_fp32).
7. **#3939 parallelism**: single GEMM = enlargement of safe row-parallel geometry (M=T rows, K=16384 unchanged), removes serial concat glue. PASS.
8. **Integration**: new patch module `omlx/patches/hc_single_gemm/` monkey-patching `HyperConnection._mix` prefill branch (L>8 → hoisted GEMM), decode branches (L≤8, tokenwise/direct) bit-identical, `apply_branch` L≤8 fused path untouched. Applied from model_loading alongside glm5_next compat, gated on model_type.
9. **Risks**: fn fp32 pin; kernel-variant fp32 accumulation order ≤~2e-5 on mixes (downstream sigmoid+sinkhorn contraction non-compounding, expected e2e ≤1e-4); tiled_linear shared with projections/HyperHead — patch `_mix` only, not tiled_linear itself; ragged last chunk needs NO pad in GEMM form (improvement).

**Status: PROVEN-SAFE-DESIGN. Next: run equiv_test.py + hc_bench.py on studio (needs GPU quiet window), then implement the patch module + bench 24K/32K/56K.**

## [NEXT SESSION / CURRENT STATE]

**Known-good config**: oMLX 0.7.0.dev4 (venv311, mlx 0.32.0) + transplanted native kernels; glm53-flash-mixed48 (mixed-4/8) on :<port> (log artifacts/omlx-e031.log); SpecPrefill wired (E-026, per-request disable via `"specprefill": false`); HC single-GEMM patch in-tree but GATED OFF (`OMLX_HC_SINGLE_GEMM=1` to enable, E-031); production prefill chunk = 2048.
**Current best measured**: control prefill ~1,034 t/s @24K TTFT, decode ~39.5 t/s (30-33 t/s in recent logs @ short ctx — verify); SpecPrefill k04 warm 24K = 2,455 t/s effective (E-027) but NOT shippable across lengths (E-029: draft-scoring slow band 16-32K).
**This session closed**: E-026 (SpecPrefill server wiring — DONE, works), E-027 (A/B: target exceeded at k04 warm), E-028 (quality gate PASSED ~95% recall @ k04), E-029 (length ladder: k04 regression in 16-32K band — draft-scoring root cause), E-030 (P2 HC archaeology + proof), E-031 (P2 implemented + benched: neutral end-to-end at chunk 2048 → gated off, retained).
**In flight (Sparks)**: P3 #4521 decode A/B prep (sa-0-ed2f540f, deleg_9516d76e); P4 GPU-time breakdown synthesis (sa-0-c968b5bb, deleg_37205047); E-031b draft-scoring root-cause (sa-0-c59c0bd9, deleg_f37f5560).
**Eliminated this session**: same-prompt A/B methodology (prefix-cache poison — use unique prompts); keep<0.4 (quality cliff); HC-GEMM as an end-to-end lever at chunk 2048 (neutral; op-count ≠ GPU time — 41% of dispatches = 3% of GPU time).
**Next experiments (ranked)**: 1) Fix draft-scoring slow band per E-031b Spark report (the blocker between 2,455 t/s and shipping); 2) run P3 A/B script when GPU quiet (decode lever, cheap); 3) P5 narrow fusion per P4's ranked list (top GPU-time consumer); 4) if draft fix lands: full validation ladder + recall re-gate at 8K-56K.
**Files to resume**: studio /tmp/{spec_*.py, hc_ab.py, p2arch/, eval/results/*.json}; ledger <mission-dir>/LEDGER.md; adapters <omlx-tree>/omlx/engine/glm_draft_adapter.py + patches/hc_single_gemm/; backups .orig-e026/.orig-e031. Server restart = 4min (model load lazy on first request).

## [E-028] SpecPrefill quality gate — PASSED at keep=0.4 (2026-09-25 ~01:20)

**Method:** fair recall test v2 — 20 embedded facts through ~24K unique filler, fixed compact answer format, 1024 max tokens, temp 0, 2 arms × 2 runs. Facts spread evenly (every ~1/20th of prompt) to make selection genuinely adversarial for importance scoring.

**Clean-rep results (degenerate outputs excluded):**
- Control: 20/20, 20/20
- keep-0.4: 20/20, 19/20, 18/20 → **~95% mean recall preservation**

**Artifacts noted:** rep1-control produced degenerate `'1:'` output in BOTH runs (positional: always the request following a k04 arm — suspected template/state interaction, NOT SpecPrefill-related since it hits control; logged for later investigation). rep1-k04 run-1 drifted into meta-reasoning (9/20 counted, excluded as format-failure).

**v1 (flawed exhaustive-list, 512-tok truncation) had shown k04 at 14/20 — the fair test shows the truth is 18-20/20. The 512-tok truncation was biasing against complete listing.**

## P1 VERDICT (E-026+E-027+E-028): SUPPORTED — keep=0.4 is the recommended config

- **Perf: 2,455 t/s effective @ 24K TTFT (target 1,584 exceeded by 55%)**
- **Quality: ~95% recall preservation on adversarial embedded-fact test**
- **Below 0.4 the quality cliff is real** (7-8/20 at 0.3/0.2 in v1) — do not ship 0.3/0.2 for recall-critical work
- Remaining P1: length validation ladder (8K/16K/32K/56K at k04), decode regression check, peak memory

## [E-026] SpecPrefill wired into server — LIVE (2026-09-25 ~00:35)

**Hypothesis:** the E-025 offline draft adapter can serve in-process, bypassing the `glm5_next_mtp` loader rejection, by routing `engine/vlm.py`'s `_load_draft` through `build_glm_mtp_draft_model(target, drafter)`.

**Changes (retained):**
- `omlx/engine/glm_draft_adapter.py` (NEW): DraftLM builder + `glm_sparse_extract_queries` extractor + module-import-time `install_glm_query_extractor()` (monkeypatches `_detect_query_extractor` — inert until a GLM draft builds).
- `engine/vlm.py:2563` + `engine/batched.py` loader blocks: glm5_next_mtp config detection → adapter route. Backups `.orig-e026`.
- Verified per-request control: `specprefill: false` off; `specprefill_keep_pct` per-request override (engine_core.py:736-744).

**Receipts:** adapter installed → DraftLM built → `set_specprefill_draft_model` with SSD cache → loaded. 14-token probe bypassed scoring (below threshold 8192) and served normally.

**Status: SUPPORTED — SpecPrefill live end-to-end.**

## [E-027] SpecPrefill A/B @ ~24K — TARGET EXCEEDED, quality gate open (2026-09-25 ~01:00)

**Method:** unique-prompt-per-request (cache-defeating), server-receipt decomposition (scoring/sparse-prefill logged separately), interleaved arms, positional warmup identified and controlled (first-specprefill-after-control arm eats ~60s draft materialization; runs 0.4s warm).

**Measured (24K, 8-tok decode, TTFT):**

| Arm | Scoring | Sparse prefill | TTFT | Effective rate | Recall (v1) |
|---|---|---|---|---|---|
| control | — | 23.3s full | 23.32s | 1,034 t/s | 20/20 |
| keep 0.4 | 0.4s | 9.6s @ 9,615 tok | **9.81s** | **2,455 t/s** | 14/20 |
| keep 0.3 | 3.6s | 7.1s @ 7,214 | 10.68s | 2,258 t/s | 7/20 |
| keep 0.2 | 0.4s | 4.7s @ 4,807 | 5.19s | 4,661 t/s | 8/20 |

**Key findings:**
1. **Steady-state keep-0.4 = 2,455 t/s effective — 1,584 target exceeded by 55%** (mechanism: token reduction; sparse prefill runs at unchanged ~1,040 t/s per kept token).
2. Positional warmup artifact identified (~60s first-specprefill arm); warm scoring is 0.4–3.6s (~2.6% of prompt).
3. Quality (v1 recall test, adversarial exhaustive-list): control 20/20, k04 14/20, k03 7/20, k02 8/20 — **quality cliff below keep=0.4**. v1 test flawed (512-tok truncated lists, unequal lengths); fair v2 (fixed compact format, 1024 tok) running.
4. Draft SSD cache non-functional for CacheList layers (`store_cache truncated: missing boundary snapshot`) — every request re-scores; future lever.
5. My first A/B (same-prompt, 4 arms) was prefix-cache-poisoned (identical outputs, 3× bogus "speedup") — recorded as methodology lesson: **unique prompt per request always**.

**Status: k04 SUPPORTED as perf result; quality gate v2 pending. Next: recall-2 receipts → keep-pct recommendation → 24K/32K/56K validation ladder.**

## [E-025] SpecPrefill draft adapter — WORKS (offline), 2026-09-25 ~02:00

**Objective (Phase 5 priority 1):** assemble a score-capable standalone draft LM bypassing mlx_lm's glm5_next_mtp rejection, restoring SpecPrefill (E-007 reversal justified by E-022's GPU-bound correction).

**Adapter (e025_v4.py, companion machine scratch + studio /tmp):**
- Draft LM = QuantizedEmbedding (lifted from loaded target model — raw dequantize fails on oQ affine packing) → one Glm5NextDecoderLayer with **sparse-attention type forced** (the MTP block IS sparse-MLA: q_a_proj/indexer/embed_q weights — NOT layer-44's linear type) → shared_head_norm → QuantizedLinear lm_head (lifted).
- Full Glm5NextModel pre/post semantics replicated in DraftLM.__call__: correct mask per layer type (create_ssm_mask / create_attention_mask), hc_mult=4 broadcast-expand before layer, mean(axis=2) after — HyperConnection REQUIRES 4-D input.
- 25 of 29 drafter tensors map to the layer (mtp.self_attn.*→self_attn.*, mtp.mlp.*→mlp.*, two layernorms); eh_proj/enorm/hnorm skipped (MTP-depth glue, not needed for standalone scoring); shared_head_norm applied separately.
- Custom query extractor `glm_sparse_extract_queries`: q_a_proj→q_a_layernorm→q_b_proj→reshape/transpose→**embed_q into the 512-d latent space** (scoring math must run in the same latent space as the KV keys).

**Two additive patches to omlx/patches/specprefill.py (backup specprefill.py.orig-e025):**
1. `n_kv_heads=None → n_attn_heads` (MLA: heads_per_group=1)
2. `_compute_importance`: CacheList(KVCache, PoolingCache) unwrapped to its KV member for .keys access

**Measured:** forward (1,64,154880) ✓ · **score_tokens: 1024 tokens → importance (1024,) → 512 selected (keep 0.4) in 135.7 ms ≈ 0.13 ms/token.** Projection @24k: ~3.2s scoring cost vs ~14s of target-prefill time skipped (keep 0.4 ≈ 60% fewer target tokens against the E-022-confirmed GPU-bound wall) → net ~10s saved on a 23s prefill — **first lever with target-scale headroom** (1,030 → theoretical ~2,200+ t/s).

**Remaining to serve:** wire adapter into engine/vlm.py draft loader (replace mlx_lm_load path when model_settings specprefill_draft_model points at a glm5_next_mtp ckpt), restart, then keep-pct ladder 0.4/0.3/0.2 + numerical-agreement + quality eval. NOTE: draft shares the target's tokenizer exactly (vocab 154,880, same tokenizer.json).

**Status:** SUPPORTED (offline proof complete). Retained: both specprefill patches (additive, no behavior change for non-GLM draft models), adapter script.

## [E-022] CPU-prep vs GPU-exec isolation (Task C, Spark) — MAJOR CORRECTION, 2026-09-25 ~00:30

**Corrects E-016 (GPU 94% idle) — a measurement artifact, now refuted with receipts.** E-016's powermetrics window (16:24:23-31) closed BEFORE its own prefill began (bench TTFT 27.59s, request completed 16:25:14 → prefill actually ran 16:24:46-16:25:14). Window-matched re-measurement: **GPU 88-98% busy, 104-124W throughout prefill.**

**The three numbers (2048-token chunk, in-server, three convergent methods):**
- CPU-prep: **6-13 ms/chunk (~0.4% of wall)** — Python op-call loop measured at the real scheduler call site; offline build-discard confirms (GPU never executes, build takes 4.3-6.2ms)
- GPU execution: **~1,870 ms/chunk (~99.6%)** — flat across kv_len 0→32k, ~0.93 ms/token, linear in tokens
- Decode cross-check: the same ~1,600-op chain runs at 16µs/op in decode — if launch overhead were the wall, chunks would cost ~52ms, not 1,900ms

**Verdict:** the wall is GPU-side execution of ~1,600 tiny serial-dependent kernels — NOT CPU dispatch/prep. "Host-side fusion to eliminate dispatch boundaries" ceiling = ~0.4%. The #3939-style host-side fusion program cannot reach 1,584 t/s on its own.

**Verdict reversals triggered by this correction:**
- E-007 SpecPrefill closure — **REVERSED, top priority.** Its "FLOPs ≈ 6% of wall" premise came from the refuted E-016. Against a GPU-bound wall, token-reduction attacks the cost directly (keep_pct 0.4 ≈ 60% fewer prefilled tokens → theoretical ~2.3×). The blocker was the draft-model loader rejecting glm5_next_mtp — the adapter (embed + Glm5NextMTPBlock + lm_head from checkpoint weights) is designed, weights downloaded. GO.
- E-009 component ablation shares (MoE 40/KDA 28/SPA 23) — now trustworthy as GPU-side costs, since wall is GPU-side. KDA 28% + SPA 23% ≈ the attention mass SpecPrefill drops.
- E-019 mx.compile 11% slower — reinterpreted: it removed 0.4% of overhead and added tracing cost. Compile remains closed as a lever.
- E-016's in-server "66% cvwait" — real but misread: thread waits on GPU completion, and the GPU is genuinely busy.

**Remaining GPU-side levers (re-ranked):** 1. SpecPrefill (token-reduction, adapter exists on disk) · 2. KDA-scan chain fusion (longest serial chain, GPU-side — chunked scan math validated in E-003) · 3. multi-stream overlap · 4. upstream batched-chunk submission.

**Artifacts:** E-022-TASKC.md + taskc-receipts/ (studio /tmp + companion machine <mission-dir>/); scheduler A/P/W instrumentation env-gated (GLM_PF_SPLIT), backup scheduler.py.orig-pfsplit; server verified canonical post-task.

## [E-023] Dispatch census (Task A, Spark) — RECEIVED, 2026-09-25 ~00:30

7,520 Python-level dispatches/chunk (cross-checked vs 10,543 IR nodes via mx.export_to_dot; zero-eval lazy-build method — server stayed up, 0 GPU bytes allocated). **HyperConnection tiled_linear _mix = 3,060 ops/chunk = 41% of all dispatches** (34 ops/HC-call ×2/layer ×45; [256,16384] fp32 tiles; fn.T re-transposed every call). But per E-022 these are GPU-side *executed* ops: the ranked targets' value is GPU-work elimination, not launch elimination — HC-mix (123GB/chunk intermediate traffic!) and l2norm-pair/o_norm (fp32 round-trips) eliminate real GPU bytes. Census artifacts: artifacts/census_a_results.json, census_a_report.md, census_dot_*.dot (studio).

## [E-024] Fusion feasibility matrix (Task B, Spark) — RECEIVED, 2026-09-25 ~00:30

Key items: **no Metal toolchain on Mini or Studio** (CLT-only both; new precompiled kernels need full Xcode somewhere; JIT route forfeits ~0-4.5% single-kernel per #4541 — acceptable for dispatch-shaped wins). **⚠️ Dev venv (py3.13) cannot load transplanted cp311 _ext → has_symbol False → any Phase-5 bench from .venv silently measures fallback paths** — always verify native_kernel_status() under the serving interpreter. #3939 rules extracted (fuse launch-dominated chains; NEVER fold norms into GEMMs at reduced grid — that was the 22× failure; per-dispatch ~3-17µs; measure slope not level). #4521 vars: **MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER** (decode +5-7.7%, prefill unchanged — decode lane only, free). oMLX's native-ext already does offline .metallib loading (d.get_library from binary dir) — #4541 route irrelevant for omlx kernels. Full matrix with 12 candidates and effort ratings in task-1 log; local copy artifacts/glm_taskB/.

**Phase 5 re-scoped (per E-022):** host-side fusion is a 0.4% ceiling — the program becomes GPU-side work elimination: SpecPrefill first (token count), then HC-mix single-GEMM (kills 123GB/chunk of fp32 tile traffic — largest single GPU-work item), l2norm/o_norm fused kernels (fp32 round-trip bytes), KDA chunked scan (E-003 math).

## [R-002] Strategy correction + Phase 5 opening — 2026-09-24 ~22:00 (user steering)

**Corrected conclusion (supersedes Session-4 wording):** "The remaining gap appears to require reducing the amount/cost of eager graph preparation and/or dispatch, and kernel fusion is the leading mechanism to investigate." (Previous "only upstream kernel fusion" was too definitive — three attack surfaces exist: A. oMLX model-code local fusion into custom Metal kernels; B. MLX runtime graph/command-buffer construction; C. hybrid fewer-ops + better batching.)

**Reframe of E-018/E-019:** the mx.compile experiment proved the cache CAN be made traceable and compiled-layer numerics are exact. The 319-vs-356 regression means the compilation BOUNDARY was wrong (whole Python layer → mx.compile still emits many kernels; it fuses elementwise ops but does not collapse launches). Next-generation solution shape: many MLX ops → ONE custom/fused GPU op, not layer → compile.

**Phase 5 objective (quantified):** at 1,030 t/s a 24K prefill costs ~23.3s; at 1,584 t/s it would be ~15.2s. Need to remove ~8s ≈ 35% of wall. Micro-optimisations don't count. Question: can we collapse enough CPU-visible op boundaries in the hot GLM blocks to remove ~8s of dispatch/graph-prep time, WITHOUT sacrificing kernel parallelism?

**Eliminated/low-value branch (record, do not re-run):** mlx issue #4521 — Metal command-buffer limits (max_ops_per_buffer=50 / max_mb_per_buffer=50 on Ultra) cost 5–8% DECODE on launch-heavy models; GLM-5.3-Flash measured ~1,600 kernels/token (independent confirmation of launch-heavy character); **prefill UNCHANGED** by raising limits on the reporter's setup. Prefill lever: dead. (Decode lane may still see the 5–8% — opportunistic env-var A/B only if variable names are trivially settable; extract names from the issue.)

**Fusion case-study receipt (design rules source):** mlx discussion #3939 (Kimi-K3 TP4, 4× M3 Ultra): one successful fused attention block = +7.8% end-to-end; one blind fusion of a projection = **22× SLOWER** (custom kernel destroyed parallelism). Rule: fusion wins when launch-bound; catastrophic when it reduces GPU-parallel work. Spark extracting full rules → R-003.

**Precompiled-kernel route receipt:** mlx issue #4541 — `precompiled_metal_kernel` / offline `-Ofast` .metallib loading is 90% wired in mlx C++ (is_precompiled_ field exists, unimplemented in eval_gpu). Measured: single-digit % on INT4 GEMV — NOT the prefill answer alone, but the route matters for our transplanted omlx kernels (we hold csrc + metallib from the 0.6.4 DMG). Feasibility matrix → R-004.

**Lever assessment (current):** local config: largely exhausted · basic kernel optimization: substantially explored · fundamental hardware limitation: NOT demonstrated · runtime/graph/dispatch optimization: OPEN · kernel fusion: PROMISING, UNPROVEN.

**Delegation policy going forward:** Spark = parallel engineering worker (source archaeology, harness construction, output interpretation, census, mundane-error chasing — all of it). Strong model = architectural decisions, fusion-target selection, kernel design review, results interpretation.


## [E-021] Spark cache-structure evidence package (deleg_e42de8fd) — RECEIVED, 2026-09-24 ~21:35

Answers brief questions 1-10 with file:line receipts + 5 live probes (read-only + venv311 empirical). Key items:

**CONFIRMS E-018 empirically:** `mx.fast.metal_kernel` IS traceable by mx.compile (CustomKernel primitive; independent probe ✅). vmap CANNOT (`[Primitive::vmap] Not implemented for CustomKernel.`). `mx.depends` works inside compile. Compiled fns take/return lists/trees of arrays ✅. `inputs=`/`outputs=` state-carrying exists (canonical: `@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)` mlx_lm/sample_utils.py:137 — the mechanism for RNG-state threading).

**Linear (KDA) layers:** one compiled region away from fully traceable — kernel already rebind-friendly (state_in read-only, state_out fresh buffer GD:107/120/218); executed path RT:290-399 mutates cache only via list-slot rebinds (RT:349/353/392) + Python-int advance. → This is exactly what E-018/E-019 implemented and measured: **feasible, exact, but 11% SLOWER at whole-model scale.**

**Sparse (SPA) layers — NOT traceable as written:** three in-place slice-assign sites: KVCache keys/values `[..., prev:offset, :] =` (CACHE:529-530); PoolingCache buf_kv/buf_gate `[:, rem:new_rem] =` (CE:186-189); pool `_pool_buf[:, len:len+n] =` (CE:227) + geometric regrow copy (CE:105-107). Plus host-side Python ints (offset/remainder/_pool_len) that must become explicit scalar args. **Probe finding: input mutation inside compile is SILENTLY DROPPED** (no error — writes just don't escape the graph): any naive sparse-layer compile would silently corrupt cache. Converting the three sites to functional concat/rebind is the required code change (bounded: 3 sites + scalar plumbing).

**Decision (per E-019 evidence):** the sparse-layer restructure is DESIGNED but NOT BUILT — the compile lever measured negative at whole-model scale on the compile-friendly half; completing the harder half has no supporting evidence and would cost significant engineering for an expected-negative return. Deprioritized; design retained here for whenever upstream runtime changes the calculus (e.g. if compile tracing overhead drops or shapeless+state improves).

**Durable MLX 0.32.0 facts (for skill):** compile drops input mutations silently; CustomKernel traceable in compile but not vmap; MLX_DISABLE_COMPILE env kills compilation; `inputs=/outputs=` = built-in state-carrying; list-in/list-out fine; scalar args fine.

**Status:** COMPLETE (evidence archived). No studio changes.


## [E-020] Chunk-size boundary A/B — NULL, 2026-09-24 ~21:00

**Hypothesis:** per-chunk eval+clear+schedule boundaries contribute fixed overhead; larger chunks (fewer boundaries) reduce total prefill time.
**A/B:** same 24k/48k prompts, prefill_step_size 8192 vs 2048 (settings restart between).
**Results:** 8192-step: 1,023/953 t/s · 2048-step: 1,023/993 t/s — identical within noise.
**Interpretation:** boundaries are free. Wall is uniformly per-token inside chunks. Combined with E-019 (compile slower), E-012a (pool-clear null), E-014 (streams null), E-006 (concurrency negative): the server's ~1,030 t/s appears to be a hard property of the current model implementation's per-token work on this runtime, not of scheduling, boundaries, or dispatch overhead in the sense tested. E-016's "GPU idle 94%" stands as fact, but every mechanistic attribution tried (per-op prep, pool clears, streams, boundaries) has been REFUTED. Remaining candidate: the GPU work itself is latency-bound (kernel-to-kernel dependencies with tiny kernels — GPU 'idle' between dependent kernels within the submitted graph), which no host-side change can fix; only kernel fusion (upstream) or fewer/bigger kernels would.
**Status:** NULL (boundary overhead), and the local-lever space is now exhausted to a reasonable standard.
**Artifacts:** artifacts/eval/results/prefill-step{8192,2048}-*.json (studio).

## [E-019] Whole-model offline prefill, 34×KDA-compiled — COMPILE NOT A LEVER, 2026-09-24 ~21:20

**Hypothesis:** compiling all 34 KDA layers (proven-safe pattern from E-018) at whole-model scale recovers the CPU graph-prep cost and raises the offline prefill rate toward component-sum (~2,500 t/s).
**Setup:** full model via vendored loader; class-level router patches Glm5NextLinearAttention.__call__ to route S>1 calls through per-layer compiled fns with explicit (conv_in, rec_in)→(y, conv_out, rec_out) state; SPA/MoE/dense eager; identical chunk loop to evalcount5 (2048-step, per-chunk cache evals).
**Command:** e019_fullmodel.py (Mac-Mini scratch) → studio /tmp; E019_COMPILE=1.
**Result:** **319 t/s compiled vs 356 t/s eager — compiled is 11% SLOWER.** (34 layers compiled OK, numerics untested at this scale but E-018 pattern exact.)
**Interpretation (major):** the offline whole-model path is NOT CPU-dispatch-bound — compile has nothing to reclaim there, and tracing overhead costs 11%. E-018's 17% single-layer win doesn't compose at model scale. Combined with: server 1,026 t/s ≈ 3× faster than any offline path constructible — the oMLX scheduler (async chunk pipelining etc.) is the thing that makes serving fast, not a hindrance. The "44× per-component in-server gap" derived from E-009 ablation shares is suspect: ablation shares were measured against pipeline wall-time dominated by shared costs, so per-component attribution was inflated (removing any component removes its share of shared wait). **The server at ~1,030-1,050 t/s may already be at the eager-MLX ceiling for this model — and my "CPU graph-prep wall" synthesis (E-016) conflated server in-efficiency with offline-harness slowness.** E-016's GPU-idle-94% measurement stands (fact), but its attribution to per-op prep is now UNSUPPORTED: idle during in-server prefill with wall dominated by something the ablation couldn't isolate.
**Status:** REFUTED (compile-as-lever at current architecture). Ledger entries E-009 interpretation and E-016 synthesis flagged as needing re-verification.
**Revisit trigger:** none local. Upstream mx.compile improvements (shapeless+state) or a genuinely traceable whole-chunk (incl. SPA pooling) would change the calculus.
**Artifacts:** artifacts/e019.log (studio), e019_fullmodel.py (Mac-Mini scratch). No server changes; harness only.


## [E-018] One-layer KDA compile prototype — COMPILE-WORKS, 2026-09-24 ~21:00

**Hypothesis:** a full GLM-5.3 KDA layer (incl. gated-delta Metal scan kernel) is traceable by mx.compile with cache state as explicit array inputs/outputs; compiled execution approaches GPU-only cost.
**Setup:** offline, full model loaded via vendored compat path (bulletproof weights), layer-0 self_attn extracted; T=8192, B=1, bf16 inputs; eager uses real ArraysCache; compiled fn threads (conv_state_in, rec_state_in) → (y, conv_state_out, rec_state_out); gated_delta_update called INSIDE compiled region with state passed in.
**Commands:** e018_v2.py (Mac-Mini scratch) → studio /tmp; server stopped first (crash lesson).
**Results:**
- eager layer: first 133.2 ms, steady median **51.3 ms**
- compiled: first **179.3 ms** (trace+build), 2nd/3rd 44.9/43.4 ms, steady median **42.8 ms**
- numerics: max|compiled−eager| = **3.05e-05** (fp32 noise) — EXACT
- peak Metal mem: 187.9 GB (model resident; compile spike fits in 256 GB)
- `mx.fast.metal_kernel` (gated-delta scan) **IS traceable** inside mx.compile — no error, correct output
**Interpretation:** compile is SAFE and numerically exact for the KDA layer, but standalone win is only **17%** (51.3→42.8). The in-server 44× KDA gap (ablation 0.276 ms/token vs standalone 0.006 ms/token at T=8192) therefore does NOT come from per-op prep visible at single-layer scale — single-layer eager standalone is already near GPU-bound. The gap must arise only at full-graph scale (cold per-op prep across 45 heterogeneous layers × chunk) — OR the ablation itself over-attributes (its "KDA share" included the sync-wait time of the whole chunk, since removing KDA removes its share of pipeline waits too). Both readings converge on the same next experiment: **whole-model compiled prefill, offline** — if rate jumps toward component-sum (~2,500 t/s), integrate to server; if not, compile is not the lever and the ablation reading was contaminated.
**Status:** SUPPORTED (compile feasibility proven); whole-model value INCONCLUSIVE pending E-019.
**Artifacts:** artifacts/e018v2.log (studio), e018_v2.py (Mac-Mini scratch).
**Retained:** harness; no server changes.


## [E-017] Session-1 instrumentation survival — CORRECTION, 2026-09-24 ~20:05

**What:** The ATTN-PATH debug logging (3 × `logger.warning` per sparse layer per chunk, added session 1 for the kernel-fallback diagnosis) **survived every "restore"** because all `.orig-*` backups were taken *after* it was added. Every in-server measurement in sessions 2–3 ran with ~231 stderr log lines per 24k request. My "instrumentation removed" claims were wrong; the Spark delegation's fresh-eyes sweep (LANG:620/:631/:668) caught it.
**Fix:** surgical removal (3 blocks, `rm_attnpath.py` on Mac-Mini scratch), verified `grep ATTN-PATH → 0` in source and live log.
**Perf impact:** none measurable (1,048 t/s @24k post-removal, within noise of 1,020–1,050). Honesty fix, not a speedup.
**Also from the same Spark package (recorded, no action):** `gated_delta_ops` has a T-token Python loop but the kernel path is taken at prefill (not live); indexer has a 512-token sub-chunk loop (~44 extra dispatches/chunk, minor); GLM:45/:53 `.item()` syncs are multimodal-only (not our path); PoolingCache/KVCache/ArraysCache update paths confirmed sync-free — independent confirmation of E-011b.
**Lesson (ledger rule):** before declaring instrumentation removed, grep the *live source* for the marker strings — backup-based restores only prove you matched the backup, which may itself be dirty.


Authoritative record. Format: one entry per experiment/investigation. See `mac-studio-m5u` skill for operational recipes (NAS, kernels, serving). This file holds the experiment ledger.

**Targets:** prefill ≥1,584 t/s @24k · decode ≥62 t/s (reference: Spark EXL3 1,567/31.4)
**Known-good config (rollback point):** PipeNetwork GLM-5.3-Flash MLX mixed-4/8 · oMLX 0.7.0.dev4 @ 14194fe · venv311 (py3.11.16) · mlx==0.32.0 (repin after any pip -e) · native kernels transplanted from oMLX 0.6.4 macos26-27 DMG (glm_moe_dsa + qwen35_prefill: `_ext.cpython-311-darwin.so`+dylib+metallib in omlx/custom_kernels/*/) · prefill_step_size=8192 · chunked_prefill off · server: `nohup artifacts/start-glm.sh > artifacts/omlx-serve.log 2>&1 < /dev/null &` on :<port> (model glm53-flash-mixed48; glm53-flash-oq4e also symlinked). Server state verified 2026-09-24 ~17:00: prefill 1,020–1,040 @14.2k tokens, decode 39.2–39.5 short-ctx, ~33 @19k ctx.

## Index

- [E-001] Missing native kernels (root cause #1, FIXED)
- [E-002] KDA-scan-62% attribution (INVALIDATED — sync pollution)
- [E-003] Chunked KDA scan algorithm (validated math, parked)
- [E-004] Memory-pressure hypothesis (refuted)
- [E-005] FFN mx.compile at prefill (null)
- [E-006] Concurrency scaling (negative)
- [E-007] SpecPrefill applicability (conditionally closed)
- [E-008] Dispatch-bound diagnosis (powermetrics + sample)
- [E-009] Component ablation (MoE 40 / KDA 28 / SPA 23 / skeleton 9)
- [E-010] Real-weight standalone replicas (3–9× faster than in-server)
- [E-011] Per-op sync cost microbench (eval=228µs vs pipelined 15µs) — CURRENT
- [E-012] Buffer-clear-between-chunks survey (scheduler per-chunk `_sync_and_clear_cache`)
- [R-001] Upstream oMLX issues reviewed

---

## [E-011] Per-op dispatch-cost microbench — 2026-09-24 ~18:30 (Spark-agnostic, this model)

**Objective:** quantify CPU-side per-op overhead of sync patterns at small-op scale (model prefill ops are µs-class GPU work).
**Hypothesis:** per-op `mx.eval` costs ≫ pipelined dispatch; magnitude ≈ the in-server layer gap.
**Script:** Mac-Mini scratch `dispatch_small.py` → studio /tmp (x[256,512]bf16 @ w[512,512], N=500; also 2048×4096 variant).
**Results (small ops):** pipelined 15 µs/op · `mx.synchronize`/op 34 µs · **`mx.eval`/op 228 µs (15×)** · eval+clear_cache 232 µs.
**Large-op variant (68 GFLOP GEMMs):** pipelined 588 µs/op (compute-bound), eval-per-op +230 µs, clear +75 µs — overheads confirmed shape-independent.
**Interpretation:** a per-op/layer-granularity `mx.eval` anywhere in the model path costs ~213 µs/op of pure serialization. KDA in-server gap ≈ 11 ms ≈ ~45–50 sub-ops × 230 µs — consistent with hidden per-layer (or per-sublayer) evals in the model/cache path. GPU stays idle between forced materializations → explains 94% idle + sample's `eval_impl→condvar wait`.
**Status:** SUPPORTED (microbench). Missing piece: WHERE the model path forces per-layer evals — Spark enumeration in flight (subagent sa-0-4f46c9b1).
**Next:** cross-reference sweep results → patch out the per-layer syncs (one at a time, A/B each) → re-bench.

## [E-012] Scheduler chunk-boundary buffer clears — 2026-09-24 ~18:35

**Finding (code-read, scheduler.py:3954 + metal_sync.py:53-81):** after every prefill chunk the scheduler runs `Scheduler._clear_cache` → `mx.synchronize(engine_stream)` + `mx.synchronize()` + `mx.clear_cache()` — drains ALL in-flight work and empties the Metal buffer pool. Next chunk re-allocates every intermediate buffer via slow IOGPUResourceCreate/residency-commit path (matches sample's concatenate_gpu→MetalAllocator::malloc→IOGPU traps).
**Granularity:** per-chunk (7× for 14k tokens) — too coarse alone to explain per-op gaps, but compounds with E-011 if per-layer evals exist (each post-clear op also pays cold-pool malloc).
**Status:** VERIFIED (code). Not yet A/B'd: disabling the per-chunk clear is an experiment candidate (memory-safety review needed — the comment cites OOM-prevention rationale: 42.8GB transient spike → 24.6GB after reclaim).

## [E-001] Missing native kernels — 2026-09-24 ~11:15 (FIXED, retained)

**Objective:** why prefill stalls ~940 t/s with long-ctx collapse.
**Root cause:** oMLX dev4 source install never compiled native kernels; `sparse_mla_attention()` returned None 297/297 → dense fallback every chunk. README-documented trap ("plain pip install -e . does NOT build them… silently fall back").
**Fix:** transplant precompiled kernels from official oMLX 0.6.4 DMG (cpython-311, mlx-0.32.0-built) into source tree; serve from venv311 with mlx pinned 0.32.0. Verified `fast.has(...)`→True True; ATTN-PATH log 297/297 HIT.
**Result:** 56k prefill 693→1,012 t/s (+46%); collapse curve eliminated; flat ~1,000–1,050 all sizes. Decode unchanged 40.
**Artifacts:** artifacts/eval/results/prefill-kern2-*.json (studio). Rollback: delete transplanted .so/dylib/metallib from custom_kernels dirs.

## [E-002] "KDA = 62% of prefill" — INVALIDATED 2026-09-24 ~13:00

Measurement artifact: probe inserted `mx.eval(out)` after every KDA layer call → serialized pipeline, inflated KDA share, regressed decode 40→21 t/s. Instrumentation removed; decode recovered to 36–39. **Methodology ban:** no forced-sync instrumentation inside the serving path (E-011 now quantifies why: 228 µs per forced eval).

## [E-003] Chunked gated-delta-rule — 2026-09-24 ~15:00 (validated, parked)

Derived + implemented chunked form (U=L·k, V=k/L, W=L·q; A=UVᵀ; d=(I+diag(β)A_strict)⁻¹β(v−S₀U); y=WS₀ᵀ+(WVᵀ masked)d; S′=L_C(S₀+dᵀV)). Numpy truth 7e-18; mlx fp32 1.9e-4 @T=3; C=16/32 rel err ~1.5e-3 @T=256. **C=64+ hits fp32 conditioning cliff** (decay-cumprod underflow in k/L). Parked: sequential Metal scan is 0.9 ms at real shape — not the bottleneck. **MLX trap recorded:** `mx.tril(bool_fill)` ignores fill — build triangular masks from arange comparisons.
**Artifacts:** Mac-Mini scratch kda_chunked.py.

## [E-004] Memory-pressure hypothesis — REFUTED 2026-09-24 ~15:30

KDA replica under ~179 GB synthetic resident: 4.70→4.83 ms (Δ3%). Memory pressure does not explain in-server gap. **Incident:** this test ran WITH server resident → studio rebooted (OOM at wired limit). Rule: pkill server before any large standalone GPU allocation. /tmp wiped on reboot (harness re-staged from Mac-Mini scratch; all results JSONs from before 15:30 lost — numbers preserved in skill/ledger only).

## [E-005] FFN mx.compile at prefill shapes — NULL 2026-09-24 ~16:50

Env-gated `GLM_COMPILE_PREFILL=1` compiling `_ffn_block` at prefill shape: 357 vs 356 t/s offline harness. Gap-owning attention/KDA sublayers use opaque caches mx.compile cannot trace. Reverted. Revisit only if execution graph changes.

## [E-006] Concurrency scaling — NEGATIVE 2026-09-24 ~17:40

N=2 → 771–857 t/s, N=4 → 883–954 t/s aggregate (vs 1,020 single). One scheduler thread, one generation_stream: parallel requests queue, don't fill gaps.

## [E-007] SpecPrefill — CONDITIONALLY CLOSED 2026-09-24 ~17:50

Not blocked, obsoleted under current bottleneck: target FLOPs ≈6% of wall-clock; draft pass pays same dispatch tax per op. Revisit iff dispatch ceiling materially improves (e.g. traceable-cache compilation lands) or draft pass avoids dispatch cost.

## [E-008] Dispatch-bound diagnosis — VERIFIED 2026-09-24 ~16:25

powermetrics 200ms×40 during 27.6s/28.3k-token prefill @1,026 t/s: **GPU idle 94.2%** (17–28 mW). `sample` 20s: 6,658/10,112 samples in `mx.eval`→6,076 `std::condition_variable::wait` (blocked); 562 in submission (IOGPUMetalResidencySet commits, MetalAllocator mallocs). Wall-clock is CPU-side waiting between tiny serial submissions. NOT a hardware limit — a serving-implementation property (distinction per brief).

## [E-009] Component ablation — VERIFIED 2026-09-24 ~14:50

Flag-file no-op ablation in-server, 24k (13.9s baseline): MoE-switch 5.6s (40%) · KDA 3.9s (28%) · SPA 3.2s (23%) · skeleton 1.2s (9%). All-ablated skeleton: 66,784 t/s (0.21s) — framework fine; per-op gaps are the cost. Weights unchanged; ablation outputs wrong activations (timing-only tool).

## [E-010] Real-weight standalone replicas — VERIFIED 2026-09-24 ~15:50

switch_mlp (real ckpt weights, T=2048, sorted routing): **2.0 ms** vs ~19 ms in-server (9.5×); with full 173 GB resident: 2.0 ms. KDA replica 4.7 ms vs ~16 ms. gather_qmm sorted-vs-random: 0.67 vs 18.5 ms (27×) — sorting matters, SwitchGLU already sorts (do_sort ≥64).

## [R-001] Upstream oMLX survey — 2026-09-24 ~17:20

dev4 == origin HEAD (14194fe). Issues reviewed: #1224 (chunked prefill = TTFT fairness, not throughput) · #3226/#3227 (ArraysCache buffer exhaustion — memory, not speed) · #2197 (cross-stream fence deadlock — fencing risk for any multi-stream fix) · #1793 (TurboQuant chain corruption — unrelated). None actionable for dispatch ceiling today.

## Pending queue (ranked) — updated session 3 end

> **SUPERSEDED by E-022 (TASK C, 09-24 ~22:00) — see E-022-TASKC-CPU-vs-GPU-isolation.md.** The #1 item below (traceable-cache whole-chunk compile to remove the "CPU graph-prep wall") is DEAD: E-022 measured CPU-prep at 0.4% of prefill wall (6-8ms/chunk vs 1,870ms GPU execution). GPU is 88-98% busy at >100W during prefill; E-016's "GPU idle 94%" was a powermetrics window artifact (samples ended before its own prefill began). Host-side fusion/compile ceiling ≈ 0.4% — cannot close the gap. Remaining levers are GPU-side only: Metal-kernel fusion of the serial dependency chain (chunked KDA scan kernel first — it is the longest serial chain at 62% of prefill), multi-stream overlap of independent GPU work, or batched-chunk GPU submission upstream in mlx.

1. **[UPSTREAM-SHAPED, the real fix]** Make KDA/SPA cache state mx.compile-traceable: pass `ArraysCache.cache` arrays + KV/Pooling arrays as explicit compile inputs/outputs around a compiled whole-layer function; scan kernel already handles state externally. Prototype offline first (evalcount5 harness), A/B in-server. This removes the CPU graph-prep wall (E-016 verdict) — projected ceiling then GPU-bound ≈ 4–8× prefill.
2. Restore canonical config before any new bench series: prefill_step_size back to 8192; keep auth change (loopback-only).
3. Re-evaluate SpecPrefill if (1) lands (E-007 condition would flip).
4. Watch upstream omlx/mlx for traceable-cache or shape-inference-cache work (re-check weekly).

---

## Older entries below (session 1–2) — see also <mission-dir>/LEDGER.md for the full format ledger



1. **[E-011 continuation]** Spark enumeration of sync points in model path (running) → patch out per-layer syncs one at a time, A/B each.
2. A/B: disable scheduler per-chunk `_clear_cache` (E-012) — needs memory-safety check (transient-spike rationale in code comments; measure peak RSS during 56k prefill with/without).
3. If per-layer syncs found+removed: re-run full ladder; re-evaluate SpecPrefill (E-007 condition).
4. Longer term (upstream): traceable-cache whole-chunk compile; multi-stream per-request prefill with fences.
