# Step-by-Step Setup — HW L6: Quantization + Serving (BF16 vs on-the-fly 8-bit)

**Course:** Nebius Academy · Performance Engineering · Project `tlv-pe-hw3`
**Notebook:** `hw_l6_quant_serving.ipynb`
**Model:** `Qwen/Qwen2.5-7B-Instruct` (dense, ungated, ~7.6B — fits H100 and L40S)

---

## ✅ GPU box is provisioned and ready (H100 @ <GPU_HOST>)

The Nebius H100 VM is set up end-to-end. **Connect with:**
```bash
ssh -i ~/.ssh/nebius_ed25519 <SSH_USER>@<GPU_HOST>
```
Done on the box (Sections B–D below already complete):
- **Hardware:** NVIDIA H100 80GB HBM3, compute cap **9.0** → FP8 path. 215 GB free disk, 16 CPU, 200 GB RAM.
- **Project dir:** `~/hw3/` — contains `hw_l6_quant_serving.ipynb`, a **real** `requirements.txt`, and `.venv`.
- **Env:** `~/hw3/.venv` with `torch 2.11.0+cu130` (CUDA OK), `vllm 0.25.1`, `guidellm 0.7.1`, `matplotlib 3.9.2`.
- **⚠️ Extra fix applied — `python3-dev`:** the box was missing `Python.h`, which made `vllm serve`
  crash at load (`fatal error: Python.h: No such file or directory` during a JIT compile). Fixed with
  `sudo apt-get install -y python3-dev python3.12-dev`. **If you rebuild the VM, re-run that.**
- **Smoke-test PASSED:** `vllm serve Qwen/Qwen2.5-7B-Instruct --quantization fp8 --max-model-len 4096`
  reached `/health` 200 + `Application startup complete`; log shows `quantization=fp8` and
  `Selected CutlassFP8ScaledMMLinearKernel` (FP8 genuinely active). Server was stopped; GPU is back to 0 MiB.
- The Qwen2.5-7B weights are now **cached** on the box, so the Q2 official run won't re-download.

**Sections B, C, D, E are DONE.** The full notebook was executed headlessly on the H100
(`jupyter nbconvert --execute --inplace`) with **no cell errors**; the executed notebook and all
artifacts were pulled back to this project (`hw_l6_quant_serving.ipynb` + `results/q1/`, `results/q2/`).

### Q2 measured results (H100, BF16 vs FP8)

| metric | BF16 | FP8 | FP8/BF16 |
|--------|-----:|----:|:--------:|
| weight memory (GiB) † | **14.29** | **8.2** | **0.57× (1.74× smaller)** |
| peak throughput (tok/s) | 5854 | 8075 | **1.38×** |
| requests/s (saturated) | 23.5 | 32.0 | 1.36× |
| TTFT median (ms) | 21.5 | 24.8 | 1.15× (prefill ~flat) |
| TTFT p99 (ms) | 50.1 | 44.3 | 0.88× |
| ITL median (ms) | 5.94 | 4.12 | **0.69× (decode faster)** |
| ITL p99 (ms) | 5.96 | 4.13 | 0.69× |
| req latency median (s) | 1.54 | 1.08 | 0.70× |

† **Gotcha:** the `DO-NOT-EDIT` harness prints `weights = NaN GiB` because its log regex expects
`"weights took X GiB"`, but vLLM **0.25.1** logs `"Model loading took X GiB"`. The real numbers above
come from `results/q2/vllm_bf16.log` / `vllm_fp8.log` (`Model loading took 14.29 / 8.2 GiB`). Cite
those in Q2.1; don't edit the harness cell.

**Only remaining work: the Q2.1–Q2.4 writeup** (final markdown cell) — reason from the numbers above.

### F · Shutdown
**Stop the VM in the Nebius console** when done so you're not billing idle GPU (GPU is at 0 MiB now,
but the instance is still running).

To work in Jupyter: `cd ~/hw3 && . .venv/bin/activate && jupyter lab --no-browser --port 8888`
then port-forward from your Mac: `ssh -i ~/.ssh/nebius_ed25519 -N -L 8888:localhost:8888 <SSH_USER>@<GPU_HOST>`.

---

## ✅ What's already done (by Claude)

All the code you needed to write is written and validated off-GPU:

| Item | Status | Notes |
|------|--------|-------|
| `asgmnt_plan.md` Phase 1 | ✅ updated | Now uses the Nebius **Notebook service** (`tlv-pe-hw3`), not k8s/registry/SkyPilot (that's HW2). |
| **Q1** `quantize_dequantize` | ✅ implemented | Symmetric/asymmetric, per-tensor/per-channel; guards the all-zeros scale. |
| **Q1** `quantization_error` | ✅ implemented | MSE, max-abs-err, SNR(dB) with `inf` on mse==0. |
| **Q2** `MODEL` pick | ✅ set | `Qwen/Qwen2.5-7B-Instruct`. |
| **Q2** `vllm_launch_args` | ✅ implemented | Adds `--quantization QUANT_METHOD` only when `quantize=True`; sets port/max-model-len/gpu-mem-util 0.90. |
| **Q2** `guidellm_command` | ✅ implemented | guidellm **0.7.1** spec syntax, **verified against the pinned version's real schema** (see below). |
| **Q2** `extract_metrics` | ✅ implemented | Parses `benchmarks[i].metrics[name].successful[stat]`. |
| `hw_l6_quant_serving_ilay_nisim_boris_kogan.md` mirror | ✅ regenerated | Kept in sync with the notebook. |

**Validation performed (off-GPU):**
- Ran the **Q1 self-check** and the **Q2 CPU self-check** via `uv run --with torch` → both printed `All checks passed`.
- Q1 numbers are sane: 8-bit per-channel SNR 42.6 dB vs per-tensor 18.1 dB (outliers isolated, as intended).
- The guidellm self-check only checks substrings, so I separately confirmed the flags against **`guidellm==0.7.1`** itself:
  - `--backend kind=openai_http,target=…,model=…` → `OpenAIHTTPBackendArgs` has both `target` and `model` fields ✅
  - `--profile kind=throughput,max_concurrency=…` → `ThroughputProfileArgs.max_concurrency` exists ✅
  - kinds `openai_http` / `synthetic_text` / `max_duration` / `synchronous` / `throughput` all valid in `guidellm run --help` ✅
  - (there is **no** top-level `--model` flag in 0.7.1 — model must live in the `--backend` spec, which is what the code does)

**What can only be proven on the GPU** (inherent to the assignment, not a code gap):
- The model actually loading + quantizing on your hardware (the smoke-test, step D).
- The real Q2 latency/throughput/memory numbers.

---

## A · Q1 locally on your Mac (no GPU, ~5 min) — optional, do it now

1. In the project folder: `uv venv && source .venv/bin/activate`
2. `uv pip install torch matplotlib jupyterlab ipykernel`
3. `jupyter lab hw_l6_quant_serving.ipynb`
4. Run cells top→bottom through the **Q1 self-check** → confirm `All checks passed`.
5. Run the **Q1 official cell** → writes `results/q1/snr_vs_bits.png` + `q1_results.json`. **Q1 done.**

## B · Provision the GPU (Nebius Notebook service)

6. Nebius console → project **`tlv-pe-hw3`** → create a **Notebook** with an **H100** flavor
   (L40S = fallback). H100 gives the native **FP8** path the writeup is framed around.
7. In the Jupyter file browser, **upload** `hw_l6_quant_serving.ipynb`. (No scp/git needed — the
   Notebook's disk persists.)

## C · Environment on the GPU box (Jupyter terminal)

8. **Fix the fake requirements file first** — the repo's `requirements.txt` is secretly a PDF
   (`pip install -r` would fail). Recreate a real one:
   ```bash
   printf 'vllm>=0.24.0\nguidellm==0.7.1\nmatplotlib==3.9.2\nnumpy>=1.26\njupyterlab==4.2.5\nipykernel==6.29.5\n' > requirements.txt
   ```
9. `python3 -m venv .venv && source .venv/bin/activate && pip install -U pip`
10. `pip install -r requirements.txt` — vLLM pulls its own compatible torch; **do not pin torch.**
11. Sanity check:
    ```bash
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
    ```

## D · Smoke-test before the paid run (~30 s, saves a 15-min failure)

12. Confirm the model loads + quantizes on your GPU:
    ```bash
    vllm serve Qwen/Qwen2.5-7B-Instruct --quantization fp8 --max-model-len 4096
    ```
    (use `int8_per_channel_weight_only` instead on an L40S). Wait for `server is up`, then **Ctrl-C**.

## E · Q2 official run

13. In the notebook (running on the GPU), run through the **Q2 self-check** → `All checks passed`.
14. Run the **Q2 official cell** — serves BF16 then 8-bit, runs synchronous + throughput benchmarks
    each (~10–15 min). Writes `results/q2/comparison.json`, `bf16_vs_fp8.png`, per-run `.json`/`.html`.
15. Run the **Q2 analysis cell** (p90/p99 tail latencies).

## F · Writeup + shutdown

16. Answer **Q2.1–Q2.4** in the final markdown cell **from your measured numbers**
    (memory ratio; prefill/TTFT vs decode/ITL; accuracy risk; one dashboard metric + one alt engine).
17. Download `results/` + the executed notebook off the VM.
18. **Stop the Notebook/instance** so you're not billing idle GPU.

---

## Quick reference

- **Quant method is auto-picked** by the harness from the GPU: H100 (cc ≥ 9.0) → `fp8`;
  L40S/older → `int8_per_channel_weight_only`. You don't set it manually.
- **Workload** (fixed in harness): 512 prompt tokens, 256 output tokens, 30 s max per run.
- **If you edit the notebook**, regenerate the mirror: `/ipynb2md hw_l6_quant_serving_ilay_nisim_boris_kogan.ipynb`.
- **Verify flags on the box** anytime with `guidellm run --help` and `vllm serve --help`.
