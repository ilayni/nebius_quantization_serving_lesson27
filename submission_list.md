# Submission list — HW L6: Quantization and Serving

What `README.md` asks for, in its own words:

> 1. The notebook, executed on your GPU with all outputs visible (both self-checks pass, both official
>    runs printed, all writeups answered).
> 2. The artifacts under `results/q1/` and `results/q2/` (plots, JSON, and guidellm's per-run `.json`
>    and `.html`).

Total: **16 files, ~14 MB** (the two throughput JSONs are 13 MB of that).

## Why the notebook alone isn't enough

Worth knowing concretely, not just as README compliance: **the executed notebook contains zero image
outputs.** Both plotting cells run under `matplotlib.use("Agg")` and call `savefig(...)` then `close()`
with no `plt.show()`, so neither plot is embedded in the `.ipynb` — `snr_vs_bits.png` and
`bf16_vs_fp8.png` exist only as files. The README asks for "plots" by name.

| Deliverable | Present in the `.ipynb`? |
|---|---|
| Both plots | **No** — 0 image outputs in the file |
| 4 guidellm `.html` reports | **No** |
| 4 per-run `.json` | **No** — the notebook prints selected medians and p90/p99, not full distributions |
| vLLM logs | **No** — the 14.29 → 8.2 GiB citation becomes uncheckable |
| Printed tables, both `All checks passed`, all four writeups | Yes — all 21 text outputs |

The notebook carries the argument completely and the evidence only partially. The gap is the artifacts.

---

## Required

### The notebook

| File | Size | Why |
|------|-----:|-----|
| `hw_l6_quant_serving_ilay_nisim_boris_kogan.ipynb` | 72 KB | The submission. Executed on the H100 2026-07-25, all outputs saved. |

### `results/q1/` — Q1 artifacts

| File | Size | Why |
|------|-----:|-----|
| `snr_vs_bits.png` | 44 KB | The accuracy-vs-bit-width plot ("plots"). |
| `q1_results.json` | 2 KB | Accuracy sweep + 7B weight-memory table ("JSON"). |

### `results/q2/` — Q2 artifacts

| File | Size | Why |
|------|-----:|-----|
| `bench_bf16_synchronous.json` | 225 KB | guidellm per-run report — BF16 latency. |
| `bench_bf16_synchronous.html` | 61 KB | Same run, interactive report. |
| `bench_bf16_throughput.json` | 5.5 MB | guidellm per-run report — BF16 saturation. |
| `bench_bf16_throughput.html` | 61 KB | Same run, interactive report. |
| `bench_fp8_synchronous.json` | 283 KB | guidellm per-run report — FP8 latency. |
| `bench_fp8_synchronous.html` | 60 KB | Same run, interactive report. |
| `bench_fp8_throughput.json` | 7.4 MB | guidellm per-run report — FP8 saturation. |
| `bench_fp8_throughput.html` | 60 KB | Same run, interactive report. |
| `bf16_vs_fp8.png` | 39 KB | The four-panel BF16-vs-FP8 comparison plot. |
| `comparison.json` | 1 KB | The summary the official cell writes. |

### `results/q2/` — the two vLLM logs

| File | Size | Why |
|------|-----:|-----|
| `vllm_bf16.log` | 95 KB | **Include these.** |
| `vllm_fp8.log` | 148 KB | |

The README's list doesn't name the server logs, but the Q2.1 writeup cites them by line —
`vllm_bf16.log:32` and `vllm_fp8.log:90` — because the official cell prints `nan` for weight memory
(its regex expects `weights took X GiB`; vLLM 0.25.1 logs `Model loading took X GiB`). Without the logs
the 14.29 → 8.2 GiB figure is an unsupported claim sitting next to a `nan`. They're 240 KB. Ship them.

---

## Include, but optional

| File | Size | Why |
|------|-----:|-----|
| `results/q2/run1_archive/` | 510 KB | First run of the same config, plus a README explaining it. The Q2.2 writeup cites the across-run comparison (ITL reproduced within 0.002 ms; TTFT median gap moved 3.3 → 2.5 ms) as evidence the ITL win is real and the TTFT difference is a sampling artifact. Keeping it makes that checkable. Drop the folder only if you also drop that one sentence. |
| `results/q2/_sample_report.json` | 1 KB | Written by the harness's own CPU self-check, not by you. Harmless either way; leaving it is the honest "this is what the run produced". |

---

## Do NOT submit

Course-provided or working notes — none of it is asked for, and some of it would look odd in a graded bundle:

- `README.md`, `README.pdf`, `requirements.txt`, `requirements.pdf` — the course gave you these.
- `hw_l6_quant_serving_ilay_nisim_boris_kogan.md` — a markdown mirror of the notebook, generated for reading. Not a deliverable,
  and submitting a second copy of the notebook invites "which one is authoritative?"
- `run_results.md`, `step_by_step_setup.md`, `asgmnt_plan.md`, `tips_hw2_hw3.md` — your working notes.
- `asgmnt_feedback.md` — the review of your own submission. Definitely not.
- `.venv/`, `backups/`, `__pycache__/`, `.ipynb_checkpoints/` — if any exist locally.

---

## Bundle it

```bash
cd ~/dev/cloud/nebius/performance_engineering_benchmarking_quatization_lesson27
zip -r hw_l6_submission.zip \
  hw_l6_quant_serving_ilay_nisim_boris_kogan.ipynb \
  results/q1 results/q2 \
  -x '*/.ipynb_checkpoints/*' '*/__pycache__/*'
unzip -l hw_l6_submission.zip   # expect 20 files: 16 + the 4 in run1_archive/
```

### Pre-flight

- [x] Notebook executed top to bottom, no cell errors, all outputs saved.
- [x] Both self-checks print `All checks passed`.
- [x] Both official runs printed; Q2 numbers come from the H100 run.
- [x] All four Q2 writeups answered (Q2.1–Q2.4).
- [x] Four `bench_*.html` present and populated (verified: real embedded data, model
      `Qwen/Qwen2.5-7B-Instruct`, correct timestamp).
- [x] Cited log lines resolve to the right text in the submitted files.
- [ ] Read the writeup once in your own voice before sending.
