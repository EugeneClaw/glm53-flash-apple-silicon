# Benchmark methodology (what invalidates a result on this stack)

Every methodological trap below produced a **false result we initially
published to ourselves**. Each is listed with the failure mode and the fix.
If you reproduce any of this work, adopt the fixes before trusting any A/B.

## 1. Prompt-cache reuse poisons A/B arms

The serving stack keeps a radix/prefix cache of rendered prompts. Running the
same prompt through arms A then B makes arm B a cache hit — we measured a
bogus "3× speedup" this way (identical outputs across arms were the tell).

**Fix:** unique prompt content per request (salted corpus generator — see
`scripts/spec_ab2.py`). Never reuse a prompt across arms, ever.

## 2. Non-streaming walls hide the decode/prefill split

A non-streaming completion wall time = TTFT + decode. Arms that differ only in
prefill still differ in wall by the same amount, but "TTFT" from a non-stream
response is not TTFT.

**Fix:** always `stream: true`, record first-content-chunk time as TTFT,
compute decode tok/s from `(wall − ttft)`. Cross-check against the server's
own completion log lines (`stream_model_ttft`, tok/s) — they matched our
client-side numbers within ~0.5%.

## 3. Powermetrics sampling windows can invert a diagnosis

The single most expensive mistake of the project: a `powermetrics` sampling
run whose window **closed before the measured prefill began** returned
"GPU idle residency 94%" — and we built an entire (wrong) CPU-dispatch
theory on it. Window-matched re-measurement showed the GPU 88–98% busy.

**Fix:** when correlating system-level traces with workloads, derive the
window from the workload's own timestamps (bench receipt + server log), not
from when you *started* the sampler. Also: `sample(1)` thread state shows
`cvwait` for threads waiting on GPU completion — that is GPU-busy evidence,
not idle evidence.

## 4. Arm-ordering warmup

The first speculative-prefill request after a cold state pays one-time draft
materialization (~60 s). Arms run in a fixed order therefore confound
treatment with position.

**Fix:** warm the treatment path with a throwaway request before the first
timed arm, and/or interleave arm order.

## 5. Corpus tokenization ratio must be calibrated

"~4 chars/token" folk wisdom was 6.78 chars/token for this model's tokenizer
on English technical text. Every prompt-size claim should state *actual
prompt tokens* from the server's usage block, not characters.

## 6. Sync instrumentation changes the thing it measures

`mx.eval()` per layer forces GPU syncs that serialized the pipeline and
halved decode speed — the "62% KDA" attribution from our first profiling
pass was an artifact of the probe. Print-instrumentation in hot paths is
banned for this reason.

**Fix:** passive counters, env-gated log lines outside the measured window,
or system-level tracing (powermetrics/sample) with correct windows.

## 7. Per-call timing floors corrupt microbenches

Per-call harness floor was 0.16–0.26 ms on this machine — larger than the
kernels being measured. Any microbench of sub-millisecond ops must fit
`T(K) = c + K·s` over many K instances inside one eval and report the slope,
never a single-call delta. (Method from mlx discussion #3939.)

## 8. Fresh-vs-long-lived server state

Control measurements from a long-lived server vs a fresh restart differ in
page-cache warmth and accumulated state. We measured them as identical for
prefill throughput *after* controlling for prompt reuse — but only by
re-running the control fresh. State your server's age with your numbers.

## 9. "Control" isn't control when a feature is default-on

SpecPrefill is default-armed once the draft loads. A "control" bench that
forgets `"specprefill": false` measures the treatment. One 2,156 tok/s
"control" datapoint in our logs is actually the treatment — caught and
errata'd.

**Fix:** every benchmark request states every feature flag explicitly.

## 10. Quality gates can measure output format, not quality

A recall test with a 512/1024-token answer cap "measured" how completely the
model listed 20 facts — arms differ in verbosity, so the cap truncates them
differently. Our first quality gate was exactly this confound.

**Fix:** budget the answer generously (2048+), enforce compact formats,
score content not length, and run both arms on identical instructions.
