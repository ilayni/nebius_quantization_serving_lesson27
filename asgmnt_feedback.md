# Feedback — HW L6: Quantization and Serving

Reviewed against `README.md` (assignment brief), the executed `hw_l6_quant_serving_ilay_nisim_boris_kogan.ipynb`, and the
artifacts under `results/`. Run: Qwen2.5-7B-Instruct on an H100 80GB, BF16 vs on-the-fly FP8,
vllm 0.25.1 / guidellm 0.7.1.

> **Status:** all three critical items are fixed and the notebook was re-executed end-to-end on the H100
> (2026-07-25, no cell errors). The numbers below and in the writeup are from that second run. Nothing is
> outstanding — the remaining sections explain what was wrong and what changed, so you can defend the
> submission rather than just hand it in.

---

## What's working

Q1 is clean. The reduce-dim handling covers per-tensor and per-channel through one code path instead of
branching twice, the zero-guard on the scale is the right instinct, and the numbers land where they
should — 18.1 dB per-tensor vs 42.6 dB per-channel at 8 bits is exactly the outlier story the harness
builds the weight matrix to demonstrate.

The Q2 writeup does three things that are easy to get wrong. It notices the KV cache stayed 16-bit and
correctly excludes it from the FP8 win. It reports TTFT *regressing* instead of quietly rounding it into
the good-news column. And it flags that the p99 and the median moved in opposite directions rather than
citing only the number that supports the argument.

---

## Critical

### 1. The `.html` reports required for submission were never generated

The README's submission list asks for "guidellm's per-run `.json` **and** `.html`". `results/q2/` has four
`bench_*.json` and no HTML anywhere in the project (`find . -name "*.html"` returns nothing).

The cause is in Part 2b: `--output kind=json,path=...` emits JSON only. You had every reason to believe
otherwise, because the locked official-run cell prints `wrote ... and per-run guidellm .json/.html`
unconditionally, whether or not any HTML exists. But the grader checks the directory, not the print.

**Fix (applied):** `--output` is repeatable in guidellm 0.7.x and `html` is a registered output kind —
both confirmed against the v0.7.1 sources (`src/guidellm/benchmark/outputs/html.py` registers
`kind="html"` with a `path` field; `docs/getting-started/benchmark.md` lists `--output` as repeatable).
`guidellm_command` now emits a second `--output kind=html,path=<same stem>.html` alongside the JSON.

**Resolved.** The notebook was re-executed on the H100 (`jupyter nbconvert --execute --inplace`, ~4 min,
no cell errors), so `results/q2/` now holds all four `bench_*.html` next to the four `bench_*.json`, and
the saved command lines in the official-run output show both `--output` flags. The harness's closing
`print` is finally telling the truth.

### 2. The headline memory number does not exist in any artifact you submit

The official run prints `nan` for weight memory and `comparison.json` records `nan` for both configs. The
writeup asserts 14.29 → 8.2 GiB "from the vLLM load logs", which is correct — it is at
`results/q2/vllm_bf16.log:32` and `results/q2/vllm_fp8.log:90` — but a grader reading the notebook sees an
unsupported number sitting directly above a `nan` row. The Q1→Q2 through-line *is* the memory saving, so
this is the one number that has to be airtight.

The cause: the harness regex looks for `weights took X GiB`, while vLLM 0.25.1 logs
`Model loading took 14.29 GiB`. You already diagnosed this in `run_results.md` — it just never made it
into the notebook, which is the thing being graded.

**Fix (applied):** Q2.1 now cites both log lines by name and explains the regex mismatch in one sentence,
and states explicitly that the locked cell was left unedited on purpose. Do not patch the regex in the
`DO NOT EDIT` cell — citing the log is the correct move and costs you nothing.

### 3. The TTFT paragraph rests on a difference you cannot resolve, and the mechanism given for it is wrong for your GPU

Two separate problems in the same paragraph.

**The evidence.** Your synchronous runs contain only n=20 (BF16) and n=28 (FP8) requests, with TTFT
standard deviations of 6.13 ms and 3.75 ms — all four numbers are in the reports you saved, at
`metrics.time_to_first_token_ms.successful.count` and `.std_dev`. That puts the standard error of each
mean at 1.37 ms and 0.71 ms, so the 1.55 ms gap in means (23.74 vs 25.29) is **one** standard error of
the difference. The p99 moving the other way (49.6 → 44.7 ms) is the same signal from the other end. You
spotted that inconsistency and treated it as a parenthetical; it was actually the evidence that TTFT here
is noise-dominated.

**The re-run settled it empirically.** Across two independent runs of the identical configuration, ITL
came back within 0.002 ms and throughput within 0.1%, while the TTFT median gap wandered from 3.3 ms to
2.5 ms. That is what a real effect and a sampling artifact look like side by side. The mechanism is
sampling density: a 30-second synchronous run produces ~20 TTFT measurements (one per request) but 5100
and 7140 inter-token intervals at 256 output tokens apiece, with σ = 0.004 ms. That contrast is now the
backbone of the Q2.2 argument, and it also strengthens Q2.4 — p99 ITL is dashboard-worthy partly *because*
it's densely sampled, which TTFT demonstrably is not.

One precision note worth keeping: the median gap (2.5 ms) and the mean gap the statistics apply to
(1.55 ms) are different quantities. Quote the means when making the significance argument.

**The mechanism.** Even if the regression were real, "prefill is compute-bound, sitting on the flat part
of the roofline where FP8 buys you nothing" does not hold on an H100 — FP8 tensor cores roughly double
peak dense FLOPs versus BF16, so FP8 *raises* the roof rather than leaving prefill sitting on it. The flat-roof
argument is the correct one for the other branch of your own assignment: weight-only INT8 on an L40S
dequantizes back to 16-bit for the matmul, so a compute-bound prefill genuinely gains nothing there. At
512 prompt tokens the prefill GEMMs are also small enough that per-token activation quantization and
kernel-launch overhead matter more than peak FLOPs.

**Fix (applied):** Q2.2 rewritten — ITL still leads, TTFT is reported as flat within noise with the n and
σ shown, and the roofline argument is assigned to the branch it actually describes.

---

## Not critical

### Do the arithmetic in Q2.1 instead of gesturing at it

"Embeddings and the LM head stay in higher precision" is the right explanation, and it is checkable in two
lines. Qwen2.5-7B is 7.62B params, and 7.62B × 2 bytes = 14.19 GiB, which reproduces the
`Checkpoint size: 14.19 GiB` line in your log — that confirms the parameter count rather than assuming it.
Embeddings plus the untied LM head are 152064 × 3584 × 2 ≈ 1.09B params staying BF16 (~2.0 GiB); the
remaining ~6.5B at one byte each is ~6.1 GiB; total ~8.1 GiB against the 8.2 logged. Predicting the
measured number to within 1% is a much stronger move than naming the mechanism and leaving it there.
(Added to Q2.1.)

### Q2.1 misses the main reason a saturated server does not reach 2×

KV traffic, attention cost, and scheduler overhead are all real, but the dominant effect is that at 64
concurrent streams decode stops being a stack of matrix-vector products and becomes matrix-matrix.
Arithmetic intensity climbs with batch size, the workload drifts off the bandwidth-bound part of the
roofline, and halving the weight bytes therefore buys back less than half the time. That is the same
physics as the prefill discussion applied to the batched case, and it ties your 1.38× to your 0.69× with
one idea instead of a list. (Added to Q2.1.)

### The Q1→Q2 outlier bridge in Q2.3 is a little too tidy for FP8

E4M3 carries a per-value exponent, so it tolerates a wide dynamic range far better than INT8's single
shared linear step — Q1's "outliers stretch the scale and crush the ordinary weights" failure is much more
an INT8 story than an FP8 one. Outliers are not free either way: per-tensor FP8 still has one shared
scale, so extreme values can still push small weights toward zero. The cleaner bridge is that Q1's
*remedy* was per-channel scales, and that is precisely what the L40S path
(`int8_per_channel_weight_only`) uses and per-tensor FP8 does not. (Softened in Q2.3.)

---

## Before you submit

- [x] `kind=html` validated against the *installed* guidellm 0.7.1 (not just the docs) before spending
      the re-run, so a malformed spec couldn't fail after the model load.
- [x] Notebook re-executed on the H100, `results/q2/` now has four `bench_*.html` beside the four
      `bench_*.json`, and the saved command lines match the code.
- [x] Every measured number in the writeup, `run_results.md`, and this file recomputed from the new
      reports. The first run's synchronous reports are archived under `results/q2/run1_archive/`.
- [ ] Read the Q2 writeup once in your own voice — it is your submission and the argument should sound
      like you. The numbers are checked; the phrasing is yours to own.
- [ ] Optional: settle TTFT properly with a **separate** longer synchronous run, outside the official
      cell, reported as a side note. Don't raise `MAX_SECONDS` — it lives in the `DO NOT EDIT` harness
      cell, and editing it to win an argument about statistics would cost you the harness integrity the
      rest of this feedback is asking you to protect.
- [ ] **Stop the Nebius VM** (`<GPU_HOST>`) — it's idle at 0 MiB but still billing.
