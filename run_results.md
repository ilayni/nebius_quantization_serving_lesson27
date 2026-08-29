# Run Results — HW L6 Quantization + Serving

**Status: COMPLETE.** Q1 + Q2 executed end-to-end on a Nebius **H100 80GB** (BF16 vs on-the-fly FP8),
notebook ran with no cell errors, artifacts pulled back into this project. Q2.1–Q2.4 writeup answered
from the measured numbers below.

- **GPU:** NVIDIA H100 80GB HBM3, compute cap 9.0 → `--quantization fp8` (CutlassFP8ScaledMMLinearKernel)
- **Model:** `Qwen/Qwen2.5-7B-Instruct` (dense, ungated, ~7.6B), same id served twice
- **Stack:** vllm 0.25.1, guidellm 0.7.1, torch 2.11.0+cu130
- **Workload:** synthetic 512 prompt / 256 output tokens, 30 s max per run

---

## Q1 — quantization from first principles (CPU-deterministic)

Weight matrix 256×1024 with 4 outlier columns. Self-check passed (`All checks passed`).

| scheme | bits | SNR (dB) | MSE |
|--------|-----:|---------:|-----|
| per-tensor  | 8 | 18.1 | 2.16e-05 |
| per-channel | 8 | 42.6 | 7.57e-08 |
| per-tensor  | 4 |  5.1 | 4.26e-04 |

Takeaway: at the same bit-width, **per-channel isolates the outlier columns** (42.6 vs 18.1 dB), and
fewer bits means more error. 7B weight-memory table: fp16 ≈ 13.0 GiB, fp8 ≈ 6.5 GiB (2×), int4 ≈ 3.3 GiB (4×).

Artifacts: `results/q1/snr_vs_bits.png`, `results/q1/q1_results.json`.

---

## Q2 — BF16 vs FP8 on the H100 (measured)

| metric | BF16 | FP8 | FP8 / BF16 |
|--------|-----:|----:|:----------:|
| **weight memory (GiB)** † | **14.29** | **8.2** | **0.57× → 1.74× smaller** |
| **peak throughput (tok/s)** | 5859 | 8070 | **1.38×** |
| requests/s (saturated) | 23.5 | 32.0 | 1.36× |
| TTFT median (ms) — *prefill* | 22.1 | 24.5 | 1.11× |
| TTFT p90 (ms) | 22.9 | 25.4 | 1.11× |
| TTFT p99 (ms) | 49.6 | 44.7 | 0.90× |
| **ITL median (ms) — *decode*** | 5.95 | 4.12 | **0.69×** |
| ITL p90 (ms) | 5.947 | 4.120 | 0.69× |
| ITL p99 (ms) | 5.957 | 4.124 | 0.69× |
| req latency median (s) | 1.54 | 1.07 | 0.70× |
| KV-cache headroom (GiB) | 55.7 | 61.9 | — |

Second run of the identical configuration (2026-07-25, ~08:48–08:52 UTC); the first run's artifacts are
archived locally under `results/q2/run1_archive/` (also on the box at `~/hw3/backups/20260725-084812/`). ITL reproduced to within 0.002 ms; the TTFT
median gap moved from 3.3 ms to 2.5 ms — see below.

**Reading of the numbers:**
- **Decode (ITL) is the big winner (~31% faster)** — decode is memory-bandwidth-bound, so halving the
  weight bytes per token pays off directly. It's also the *well-sampled* metric: 5100 (BF16) / 7140 (FP8)
  inter-token intervals per run, σ = 0.004 ms.
- **Prefill (TTFT) is unchanged within noise** — only n=20 (BF16) / n=28 (FP8) requests, TTFT
  σ = 6.13 / 3.75 ms (SE 1.37 / 0.71 ms), so the 1.55 ms gap in means (23.74 vs 25.29 ms) is **one**
  standard error of the difference. Not a result. The p99 moving the *other* way (49.6 → 44.7 ms) and
  the across-run wobble say the same thing. Note FP8 does **not** leave prefill on a flat roofline on
  an H100 — FP8 tensor cores roughly double peak dense FLOPs. That argument belongs to the L40S
  weight-only INT8 path, which dequantizes back to 16-bit for the matmul.
- **Peak throughput +38%, not 2×** — at 64 concurrent streams decode becomes matrix-matrix, so
  arithmetic intensity rises and the server is no longer purely bandwidth-bound. Plus 16-bit KV-cache
  traffic, attention compute, and scheduling, none of which FP8 touches.

† **Harness gotcha:** the `DO-NOT-EDIT` Q2 cell prints `weights = NaN GiB` because its regex expects
`"weights took X GiB"`, but vLLM 0.25.1 logs `"Model loading took X GiB"`. Real values taken from
`results/q2/vllm_bf16.log:32` (`14.29 GiB`) and `vllm_fp8.log:90` (`8.2 GiB`). Harness left unedited on purpose;
the provenance is cited in the Q2.1 writeup instead.

Artifacts: `results/q2/comparison.json`, `bf16_vs_fp8.png`, `bench_{bf16,fp8}_{synchronous,throughput}.json`
**and matching `.html`**, `vllm_{bf16,fp8}.log`.

---

## Writeup (Q2.1–Q2.4)

Full prose answers are written into the final markdown cell of `hw_l6_quant_serving_ilay_nisim_boris_kogan.ipynb`
(and mirrored in `hw_l6_quant_serving_ilay_nisim_boris_kogan.md`). Summary of the argument:

- **Q2.1** — 1.74× smaller weights, 1.38× throughput; gap vs 2× is the non-quantized parts
  (embeddings/LM head/scales) + non-weight bottlenecks on a saturated server.
- **Q2.2** — decode (ITL) improved 0.69× (memory-bound, and measured thousands of times per run);
  prefill (TTFT) unchanged within noise at n=20/28, *not* because FP8 leaves prefill on a flat roof —
  H100 FP8 cores raise the roof. L40S/INT8 is weight-only and *is* the flat-roof case.
- **Q2.3** — outlier channels + no calibration hurt quality on hard inputs; invisible to a speed
  benchmark; verify with a scored eval (MMLU/GSM8K/perplexity) BF16 vs FP8 before shipping.
- **Q2.4** — dashboard p99 ITL (SLO/saturation early-warning) + throughput (capacity); alt engine
  TensorRT-LLM for max perf on a locked-in high-volume model (SGLang for shared-prefix workloads).

---

## Remaining
- [x] ~~Re-run the official Q2 cell on the GPU~~ — done 2026-07-25 08:48–08:52 UTC via
      `jupyter nbconvert --execute --inplace`, no cell errors. `guidellm_command` now emits
      `--output kind=html,path=<stem>.html` alongside the JSON, so all four required `bench_*.html`
      reports exist and the saved command lines match the code. Writeup numbers updated to this run.
- [ ] Review the Q2.1–Q2.4 wording, tweak to your voice if desired.
- [ ] Optional: `MAX_SECONDS` lives in the locked harness cell, so leave it. If you want TTFT settled
      beyond "unchanged within noise", do a separate longer synchronous run outside the official cell
      and report it as a side note.
- [ ] **Stop the Nebius VM** (`<GPU_HOST>`) — GPU idle at 0 MiB but the instance still bills.
