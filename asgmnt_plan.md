# Assignment Plan — HW L6: Quantization + Serving BF16 vs on-the-fly 8-bit

**Course:** Nebius Academy · Performance Engineering
**Notebook:** `hw_l6_quant_serving.ipynb`
**Deliverable:** the run notebook with your implementations filled in, the artifacts under
`results/q1/` and `results/q2/`, and the Q2 writeup answered from *your* measured numbers.

---

## TL;DR recommendation

**Run Q1 anywhere (your Mac CPU is fine). Run Q2 on Nebius Cloud, on an H100.**

Do **not** use Colab. The assignment hard-targets an **H100 or L40S** GPU, and free/Pro Colab
gives you a T4/L4/A100 at best — none of which match the intended hardware, and free Colab can't
even hold a 7–8B model in BF16. Since you have free GPU on Nebius, that's the correct and cheapest
path. Pick an **H100** so you exercise the native **FP8** path the writeup is framed around
(H100 → FP8; L40S → INT8). L40S also works if H100 isn't available.

---

## What the assignment actually requires

Two questions in one notebook:

| Q | Task | Hardware |
|---|------|----------|
| **Q1** | Implement `quantize_dequantize` + `quantization_error` from first principles; sweep accuracy vs bit-width; produce the 7B weight-memory table. | Pure PyTorch — **CPU is fine** (runs anywhere). |
| **Q2** | Serve the *same* ~3–8B model twice with **vLLM** (BF16, then on-the-fly 8-bit), benchmark both with **guidellm**, and quantify latency/throughput/memory. | **NVIDIA GPU required — H100 or L40S.** Official run ~10–15 min. |

You implement 5 functions; the rest is a fixed harness (`🔒 DO NOT EDIT`). Self-checks run on CPU
against a bundled sample, but **your reported Q2 numbers must come from a real GPU run.**

Pinned stack (from `requirements.txt`): `vllm>=0.24.0`, `guidellm==0.7.1`, `matplotlib==3.9.2`,
`numpy>=1.26`, `jupyterlab`, `ipykernel`. vLLM pulls its own compatible torch — **do not pin torch.**

The 8-bit method is chosen automatically by the harness from the GPU:
- **H100** (compute cap ≥ 9.0) → `--quantization fp8` (per-tensor FP8, native FP8 tensor cores)
- **L40S / older** (cc 8.9 and below) → `--quantization int8_per_channel_weight_only` (CUTLASS int8)

---

## ⚠️ Gotcha: the two support files are PDFs, not text

`README.md` and `requirements.txt` in this folder are **PDF exports** (Chromium print-to-PDF), not
plaintext — despite the extensions. `pip install -r requirements.txt` will fail against the PDF.
Before installing on the GPU box, recreate a **real** `requirements.txt`:

```text
vllm>=0.24.0
guidellm==0.7.1
matplotlib==3.9.2
numpy>=1.26
jupyterlab==4.2.5
ipykernel==6.29.5
```

(Q1-only, no vLLM: `pip install torch matplotlib jupyterlab ipykernel`.)

---

## Why Nebius over Colab (the decision)

| Factor | Colab | **Nebius (recommended)** |
|--------|-------|--------------------------|
| Target GPU (H100 / L40S) | Not offered on free/Pro (T4/L4/A100) | ✅ You can spin up exactly H100 or L40S |
| FP8 path (needs Hopper, cc ≥ 9.0) | ❌ No H100 → FP8 branch never exercised | ✅ H100 gives the real FP8 path |
| Fit a 7–8B model in BF16 (~14–16 GB weights + KV) | ❌ Free T4 = 16 GB, won't fit; A100 maybe | ✅ H100 80 GB / L40S 48 GB with headroom |
| vLLM server subprocess + long benchmark | Fragile: ephemeral session, idle timeouts, disk caps | ✅ Persistent VM + disk, full shell control |
| Cost | Pro is paid; free is too weak | ✅ Free GPU on your account |
| "Save to disk" workaround | Doesn't solve the GPU-class problem | n/a |

Colab's "save to disk" only addresses persistence — it does **nothing** about the core blocker,
which is GPU *class*. The assignment is explicitly written for H100/L40S, so Nebius wins outright.

---

## Execution plan

### Phase 0 — Q1 locally (optional, do it now, no GPU)
1. `python3 -m venv .venv && source .venv/bin/activate`
2. `pip install torch matplotlib jupyterlab ipykernel`
3. Open the notebook, implement **Part 1a** `quantize_dequantize` and **Part 1b** `quantization_error`.
   - 1a: symmetric (`scale = max|w|/qmax`, zp=0, range `[-qmax, qmax]`, `qmax = 2^(b-1)-1`);
     asymmetric affine (`[min,max] → [0, 2^b-1]` with zero-point); per-tensor (`dim=None`) vs
     per-channel (scale per slice along `per_channel_dim`, broadcast back). Return
     `{"q","scale","zero_point","w_hat"}`.
   - 1b: `mse`, `max_abs_err`, `snr_db = 10*log10(mean(w^2)/mse)` (inf when mse==0).
4. Run the Q1 self-check → must print **`All checks passed`**.
5. Run the Q1 official cell → writes `results/q1/snr_vs_bits.png` + `q1_results.json`.

### Phase 1 — provision the Nebius GPU (use the managed Notebook service)
Per the course tips: **"HW3 — a notebook service is enough to get started."** You do **not** need the
HW2 machinery (k8s cluster, registry, SkyPilot) — that's the DDP assignment. For HW3:
1. Open the **`tlv-pe-hw3`** project in the Nebius console → create a **Notebook** (managed JupyterLab)
   with a GPU flavor: **H100** (preferred, for the FP8 path) or **L40S**. Its disk already persists,
   so the HF model cache and `results/` survive restarts.
2. **Upload** `hw_l6_quant_serving.ipynb` + the corrected `requirements.txt` through the Jupyter file
   browser (no scp/git needed). Keep the boot/data disk modest — ~200 GB is plenty.
3. Open a terminal in JupyterLab for the venv + smoke-test steps below.

### Phase 2 — environment on the GPU box
1. Fresh venv: `python3 -m venv .venv && source .venv/bin/activate && pip install -U pip`
2. `pip install -r requirements.txt` (the corrected one; vLLM brings its own torch).
3. Verify: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
4. **Smoke-test your model pick (~30 s, saves a failed 15-min run):**
   `vllm serve <model-id> --quantization fp8 --max-model-len 4096`
   (use `int8_per_channel_weight_only` on L40S). Wait for "server is up", then Ctrl-C.

### Phase 3 — implement Q2 (the 3 functions)
- **2a `vllm_launch_args`** — argv after `vllm serve`: `[model, ...]`; add `--quantization QUANT_METHOD`
  **only when `quantize=True`**; also set `--port`, `--max-model-len`, `--gpu-memory-utilization`
  (leave headroom). No `--disable-log-requests` (removed in current vLLM).
- **2b `guidellm_command`** — argv for **`guidellm run`** (0.7.x, spec-based flags):
  OpenAI-HTTP backend at `target`+`model`; synthetic `--data` from `prompt_tokens`/`output_tokens`;
  `--profile` = `synchronous` (latency) or `throughput` (needs `max_concurrency`); a max-duration
  `seconds` constraint; JSON report to `out_path`; non-interactive console. Confirm exact flags via
  `guidellm run --help`.
- **2c `extract_metrics`** — parse `report["benchmarks"][i]["metrics"][name]["successful"][stat]`;
  return one dict per benchmark with `ttft_ms_median, itl_ms_median, output_toks_per_s_mean,
  req_latency_s_median, rps_mean`.
- Run the Q2 self-check (CPU sample) → **`All checks passed`**.

### Phase 4 — Q2 official run (GPU)
1. Set `MODEL` to an **ungated, dense, ~3–8B instruct** model vLLM can quantize on the fly.
   Good pick: **`Qwen/Qwen2.5-7B-Instruct`** (dense, ungated, fits both GPUs). Avoid gated
   (Llama/Gemma) and MoE/exotic architectures.
2. Run the official Q2 cell — serves BF16 then 8-bit, runs synchronous+throughput benchmarks each
   (~10–15 min). Writes `results/q2/comparison.json`, `bf16_vs_fp8.png`, per-run `.json`/`.html`.
3. Run the Q2 analysis cell (tail p90/p99).

### Phase 5 — writeup + save
1. Answer **Q2.1–Q2.4** from *your* measured numbers (memory/throughput, prefill vs decode,
   accuracy risk, production metrics + an alternative serving engine).
2. Pull `results/` and the executed notebook back off the VM (persistent disk / scp).
3. **Shut the GPU VM down** so you're not billing/holding the instance.

---

## Cost / time notes
- Q1: minutes, free, local.
- Q2 on Nebius: model download + two loads + four short benchmarks ≈ **15–25 min of GPU time**.
  With free GPU credits this is effectively free — just remember to **stop the instance** after.
- H100 vs L40S: pick **H100** to exercise FP8 (matches the writeup's default framing). L40S is a
  valid fallback and lets you discuss the INT8-vs-FP8 contrast the questions invite.
