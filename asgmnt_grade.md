# Grade — HW L6: Quantization and Serving

**Submission:** `hw_l6_quant_serving_ilay_nisim_boris_kogan.ipynb` + `results_ilay_nisim_boris_kogan.zip`
**Run:** Qwen/Qwen2.5-7B-Instruct, NVIDIA H100 80GB HBM3, BF16 vs on-the-fly FP8, vLLM 0.25.1 / guidellm 0.7.x
**Graded:** 2026-07-25

## Score: 98 / 100 — A

`README.md` defines no point rubric, so the weighting below is mine and is stated up front as an
assumption. Every claim in the writeup that could be checked against the submitted artifacts was
recomputed from those artifacts; none failed.

| # | Item | Weight | Awarded |
|---|------|-------:|--------:|
| 1a | `quantize_dequantize` — sym/asym × per-tensor/per-channel | 20 | 19 |
| 1b | `quantization_error` — MSE / max-abs / SNR dB | 10 | 10 |
| 1 | Q1 self-check + official run + `results/q1/` artifacts | 10 | 10 |
| 2a | `vllm_launch_args` | 8 | 8 |
| 2b | `guidellm_command` | 12 | 12 |
| 2c | `extract_metrics` | 8 | 8 |
| 2 | Q2 official run on GPU + `results/q2/` artifacts | 12 | 12 |
| 2.1–2.4 | Q2 writeup (5 each) | 20 | 19 |
| — | Submission completeness + harness integrity | gate | pass |
| | **Total** | **100** | **98** |

---

## Integrity check — pass

The harness was not tampered with, and the run is honest:

- `execution_count` runs **1 → 13 with no gaps, in cell order** — a clean top-to-bottom execution, not
  a patchwork of re-runs.
- Harness constants intact: `MAX_SECONDS = 30`, `PROMPT_TOKENS/OUTPUT_TOKENS = 512/256`. The command
  lines echoed in the official-run output carry `seconds=30` and `prompt_tokens=512,output_tokens=256`,
  so the code in the cell and the outputs below it agree.
- Both `vllm serve` lines use the **identical model id**; `--quantization fp8` appears on the second
  launch only. Confirmed independently in `results/q2/vllm_fp8.log:7` (`'quantization': 'fp8'`) and its
  absence from the BF16 log.
- `comparison.json` records `NaN` for weight memory because the locked regex looks for
  `weights took X GiB` while vLLM 0.25.1 logs `Model loading took X GiB`. **The right call was made**:
  the locked cell was left alone and the log lines were cited in the writeup instead. Editing a
  `DO NOT EDIT` cell to produce a prettier artifact would have cost more than the `NaN` does.
- The zip is current — its mtime post-dates every file in it, and spot-checks of
  `results/q2/comparison.json` and `results/q1/q1_results.json` are byte-identical to disk.

Submission list from `README.md` is complete: executed notebook with outputs, both self-checks printing
`All checks passed`, both official runs printed, `results/q1/` (PNG + JSON), and `results/q2/` with all
four `bench_*.json` **and** all four `bench_*.html`, plus both vLLM logs and the comparison plot.

---

## Q1 — 39 / 40

The implementation is genuinely good. Per-tensor and per-channel go through **one** code path via a
computed `reduce_dims` tuple rather than two near-duplicate branches, and `per_channel_dim % w.dim()`
handles negative dims for free. The `torch.where(scale > 0, scale, ones)` guard is the right instinct —
it's the all-zeros / constant-tensor case that the self-check actually exercises (`_z = torch.zeros(4,4)`).

The numbers land exactly where the harness is designed to put them:

| scheme | 8 bit | 4 bit |
|---|---:|---:|
| per-tensor | 18.09 dB | 5.15 dB |
| per-channel | **42.65 dB** | 17.71 dB |

24.5 dB of headroom at the same bit width is the outlier story the weight matrix was built to tell, and
per-channel at 4 bits (17.71 dB) beating per-tensor at 8 bits (18.09 dB) is nearly a wash — which is the
sharper version of the same point.

**−1:** in the asymmetric branch, `zero_point` is left as an unclamped float. Standard affine
quantization clamps it into `[qmin, qmax]` and stores it as an integer, since it has to be representable
in the same integer format as the codes. Here it's harmless — for the shifted all-positive test tensor the
zero-point lands in range anyway, and skipping the clamp is marginally *more* accurate — but it's a
deviation from the contract the docstring describes.

---

## Q2 implementation — 40 / 40

**2a.** Correct and minimal. `--quantization QUANT_METHOD` is gated on `quantize=True`, and
`--max-model-len` / `--gpu-memory-utilization 0.90` are identical across both configs, which is what makes
the comparison fair — the KV cache budget is the same in both runs, so the difference in KV capacity is
attributable to the weights and nothing else. (More on that below.)

**2b.** The hardest part of the assignment and the one most likely to be got wrong. guidellm 0.7.x moved
to the spec-based CLI (`kind=...,key=value`) and `guidellm benchmark` no longer runs benchmarks — the
submitted command uses `guidellm run` with correct `--backend` / `--data` / `--profile` / `--constraint` /
`--output` specs, and passes `max_concurrency` on the throughput profile only. The second repeatable
`--output kind=html,path=...` is what puts the four required `.html` reports on disk; the locked cell
prints `wrote ... .json/.html` unconditionally, so this is a requirement that fails silently if you trust
the print. Good catch.

**2c.** Clean. Returns one dict per benchmark, correct stat selection per key.

**Official run.** Both models loaded, four benchmarks completed, no cell errors, all artifacts written.
Headline results:

| metric | BF16 | FP8 | FP8/BF16 |
|---|---:|---:|---:|
| weight memory (GiB) | 14.29 | 8.20 | 0.57× (1.74× smaller) |
| peak throughput (tok/s) | 5859 | 8070 | **1.38×** |
| ITL median (ms) | 5.95 | 4.12 | **0.69×** |
| TTFT median (ms) | 22.1 | 24.5 | 1.11× (within noise) |

---

## Q2 writeup — 19 / 20

This is the strongest part of the submission and it is well above the level the question asks for.

**Q2.1 (5/5).** The move that makes this answer work is reconciling the logged 8.2 GiB against a
from-scratch prediction: 7.62B params × 2 B = 14.19 GiB reproduces the `Checkpoint size` line (so the
parameter count is *verified*, not assumed); embeddings + untied LM head (152064 × 3584 × 2 ≈ 1.09B params)
stay BF16 at ~2.0 GiB; the remaining ~6.5B at one byte each is ~6.1 GiB; total ~8.1 GiB vs 8.2 measured.
Predicting a measured number to within 1% is what licenses treating `Model loading took` as weight memory
in the first place — without that, the headline number is just a log line with a hopeful label on it.
(The untied-head detail is correct and specific to 7B; the smaller Qwen2.5 sizes *are* tied.)

**Q2.2 (5/5).** ITL 5.95 → 4.12 ms with p50, p90 and p99 all moving by the same 0.69× factor — the whole
distribution shifting together is the right thing to point at. The TTFT paragraph is the best work in the
submission: rather than reporting a 1.11× "regression", it goes to
`metrics.time_to_first_token_ms.successful` and shows n = 20 / 28 with σ = 6.13 / 3.75 ms, putting the
1.55 ms gap in means at roughly one standard error of the difference — then corroborates it *empirically*
across two independent runs (ITL reproduced within 0.002 ms; the TTFT median gap wandered 3.3 → 2.5 ms).
I verified both against `results/q2/run1_archive/`; they hold. Keeping the first run archived so the
cross-run claim is checkable is exactly right.

The roofline discussion is also assigned to the correct branch: FP8 on an H100 **raises** the roof
(FP8 tensor cores roughly double peak dense FLOPs), so "prefill is pinned to a flat roof" is the L40S
weight-only-INT8 story, not the H100 FP8 one. Predicting "no mechanism for prefill to improve" on the L40S
rather than a measurable TTFT regression is the more defensible claim, and the answer says why.

**Q2.3 (5/5).** Correctly resists the tidy-but-wrong bridge from Q1: E4M3's per-value exponent tolerates
dynamic range far better than a linear INT8 grid, so the outlier failure mode Q1 demonstrates is more an
INT8 story. The real bridge — Q1's *remedy* was per-channel scales, which is precisely what the L40S path
uses and per-tensor FP8 does not — is the correct one. And the central point is stated plainly: none of
this shows up anywhere in guidellm, which measures speed and not correctness, so a scored eval
(MMLU/GSM8K/perplexity) against both servers is the go/no-go, not a generation diff.

**Q2.4 (4/5).** p99 ITL + saturated throughput is a good pick, and justifying p99 ITL partly by *sampling
density* (7140 samples vs TTFT's 28) ties the dashboard choice back to the Q2.2 statistics rather than
restating a best-practices list. The TensorRT-LLM / SGLang answer is appropriately specific about when.

**−1, and it is the one real gap:** Q2.1 asserts that the dominant reason peak throughput gains 1.38×
instead of 2× is that at 64 concurrent streams decode becomes matrix-matrix and drifts off the
bandwidth-bound part of the roofline. That reasoning is correct — but it is the only load-bearing claim in
the writeup that is *argued* rather than measured, in a writeup that measures everything else. The
submission's own data brackets it: single-stream decode (`synchronous`) gains 1.43× on output tok/s
(166 → 238) while saturated decode gains 1.38× — the win shrinking as batch size rises is the predicted
signature, and it was already on the page. A `max_concurrency` sweep outside the locked cell would have
settled it outright.

---

## Corroborations available in your own artifacts (credit, not deductions)

Two independent confirmations of the headline numbers are sitting unused in the logs. Worth knowing
before a viva, because they close off the two most obvious challenges:

1. **Throughput, from vLLM's own instrument.** `vllm_bf16.log` reports `Avg generation throughput:
   ~6255 tok/s` under 64 running requests; `vllm_fp8.log` reports `~8672`. Ratio **1.39×** vs guidellm's
   1.38×. This matters because guidellm's `output_tokens_per_second` *mean* is badly skewed by ramp-up
   (mean 5859 vs median 3842, σ 6316) — reading `mean` was correct, since 2c's spec dictates it, but the
   vLLM line is the cleaner witness and it agrees.

2. **Weight memory, from the KV cache.** GPU KV cache capacity went 1,042,640 → 1,158,448 tokens
   (Δ 115,808). Qwen2.5-7B at `kv_cache_dtype=auto` costs 28 layers × 4 KV heads × 128 head_dim × 2 (K+V)
   × 2 B = 56 KiB/token, so Δ ≈ **6.18 GiB** of freed memory against a weight delta of 14.29 − 8.20 =
   **6.09 GiB** — agreement to ~1.5%. Since `--gpu-memory-utilization 0.90` and `--max-model-len 4096` were
   held identical across both runs, that extra KV capacity can only have come from the weights. It answers
   the pedantic objection that `Model loading took` isn't literally *weights*, from a completely different
   direction. (Layer/head figures are from the published model config, not the log.)

---

## Explicitly not counted against you

- **The optional longer TTFT run** (open item in `asgmnt_feedback.md`) is out of scope. The n = 20/28
  noise argument is correct and sufficient, and the only in-notebook way to get more samples is raising
  `MAX_SECONDS` inside a `DO NOT EDIT` cell. Editing the harness to win a statistics argument would have
  been the wrong trade — this is not a gap.
- **`NaN` in `comparison.json`** — harness/vLLM version mismatch, correctly diagnosed and documented
  rather than patched around.
- **Answering in a new markdown cell (30)** instead of editing the question cell (29) — this keeps the
  questions visible next to the answers. Better, not worse.

## The one place this submission is exposed

Everything checkable is checked. What can't be graded from the files is whether the standard-error
argument in Q2.2 can be defended out loud — an examiner will probe it, because it is the claim that turns
an apparent 11% TTFT regression into a non-result. Read Q2.2 once in your own voice and make sure you can
reproduce, unprompted: n, σ, standard error of the difference, and why ITL's 5100–7140 samples make it a
different class of measurement than TTFT's 20.

Housekeeping, unrelated to the grade: the Nebius VM (`<GPU_HOST>`) is still billing.
