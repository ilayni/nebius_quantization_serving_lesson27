# HW — Quantization, and Serving BF16 vs on-the-fly 8-bit (FP8 / INT8)

**Nebius Academy · Performance Engineering**

This assignment has **two questions** in one notebook:

| Q | Topic | Where it runs |
|---|-------|---------------|
| **Q1** | Quantization from first principles — implement quantize/dequantize and measure the accuracy/memory trade-off. | anywhere (torch only; CPU is fine) |
| **Q2** | Serve the **same model** at **BF16 vs on-the-fly 8-bit** with **vLLM**, benchmark both with **guidellm**, and quantify the win. The 8-bit method adapts to your GPU: **FP8 on an H100, INT8 on an L40S**. | your **H100 or L40S** |

The through-line: **Q1 shows the memory saving on paper, Q2 measures the latency/throughput win on real hardware.**
Quantization is "same model, fewer bits" — fewer bits per weight means less HBM traffic per token, which should
cut latency and lift throughput. Q1 makes you derive the memory saving; Q2 makes you prove the runtime win.

## Cell types

- 🔒 **DO NOT EDIT** — the fixed harness (data, plumbing, self-checks, official runs). Editing invalidates your results.
- ✏️ **YOUR IMPLEMENTATION** — replace `raise NotImplementedError` with your code. These are the only code cells you change.
- ✅ **SELF-CHECK** — fast asserts that must print `All checks passed` before an official run counts.

## Hardware

Q1 runs anywhere (it is pure PyTorch on a tiny tensor). **Q2 requires an NVIDIA GPU.** The on-the-fly
quantization method picks itself based on what you have:

- **H100** (compute capability 9.0) has native FP8 tensor cores → `--quantization fp8`.
- **L40S** and other Ada/older cards (8.9 and below) lack that FP8 datapath → `--quantization int8_per_channel_weight_only`.

Either way it's the same base model, quantized at load time (no calibration, no checkpoint swap). Q2's
self-checks run on CPU (they use a bundled sample), but **your reported Q2 numbers must come from your GPU run.**

## Getting started

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Each question writes its artifacts under `results/q1/` and `results/q2/`. Run top-to-bottom and keep the outputs.

# Question 1 — Quantization from first principles

Quantization stores each number in fewer bits by mapping a continuous range onto a small set of integer
levels via a **scale** (and, for asymmetric schemes, a **zero-point**). Here you implement it and measure
what it costs in accuracy and saves in memory.

## What you implement

| Part | Function | Idea |
|------|----------|------|
| 1a | `quantize_dequantize` | round to `num_bits` integers, then map back — per-tensor / per-channel, symmetric / asymmetric |
| 1b | `quantization_error` | MSE, max-abs error, and signal-to-noise ratio (dB) |

You are given a weight matrix with a few **outlier channels** — the classic reason plain per-tensor weight
quantization struggles (a handful of large values stretch the scale and crush everything else).

## The Q1 harness (DO NOT EDIT)


```python
# 🔒 DO NOT EDIT — Q1 harness: the weight matrix, constants, and plotting.
import os, json, math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Q1_DIR = os.path.join("results", "q1")
os.makedirs(Q1_DIR, exist_ok=True)

# Bit-width -> bytes helpers for common inference formats.
BITS = {"fp16/bf16": 16, "fp8": 8, "int8": 8, "int4": 4}


def make_weight_matrix(rows: int = 256, cols: int = 1024, seed: int = 0) -> torch.Tensor:
    """A realistic-ish weight tensor: mostly small values, plus a few OUTLIER columns.

    The outliers are what break naive per-tensor quantization — one big value per row
    stretches the per-tensor scale so every ordinary weight loses precision. Per-channel
    (per-column here) scales isolate the damage. Deterministic given `seed`.
    """
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, cols, generator=g) * 0.02
    outlier_cols = [7, 123, 512, 900]
    w[:, outlier_cols] += torch.randn(rows, len(outlier_cols), generator=g) * 0.5
    return w


def bytes_for(num_params: int, bits: int) -> float:
    return num_params * bits / 8.0


print("Q1 harness ready. torch", torch.__version__,
      "| device for Q1 is CPU-friendly (tiny tensor)")
```

    Q1 harness ready. torch 2.11.0+cu130 | device for Q1 is CPU-friendly (tiny tensor)


## Part 1a — `quantize_dequantize`

Implement **fake quantization**: quantize `w` to `num_bits` integers, then dequantize back to float, and
return both the integer codes and the reconstruction `w_hat`. Support the four combinations:

- **symmetric** — `scale = max(|w|) / qmax`, zero-point `0`, integer range `[-qmax, qmax]` with `qmax = 2**(num_bits-1) - 1`.
- **asymmetric (affine)** — map `[min(w), max(w)] → [0, 2**num_bits - 1]` with a `zero_point`.
- **per-tensor** — one scale for the whole tensor (`per_channel_dim=None`).
- **per-channel** — a separate scale per slice along `per_channel_dim` (broadcast back).

Return a dict `{"q": ..., "scale": ..., "zero_point": ..., "w_hat": ...}`.


```python
def quantize_dequantize(w, num_bits, symmetric=True, per_channel_dim=None):
    """Fake-quantize `w` to `num_bits` integers, then dequantize back to float.

    Support both symmetric and asymmetric (affine) schemes, and both per-tensor
    (`per_channel_dim=None`) and per-channel (a separate scale per slice along
    `per_channel_dim`, broadcast back). See Part 1a above for the exact ranges.

    Return a dict: {"q": ..., "scale": ..., "zero_point": ..., "w_hat": ...}.
    """
    w = w.float()

    # Dims to reduce over when computing scale/zero-point:
    #   per-tensor  -> reduce every dim (one scalar scale, kept as shape [1,1,...])
    #   per-channel -> reduce every dim EXCEPT per_channel_dim (one scale per slice)
    if per_channel_dim is None:
        reduce_dims = tuple(range(w.dim()))
    else:
        d = per_channel_dim % w.dim()
        reduce_dims = tuple(i for i in range(w.dim()) if i != d)

    if symmetric:
        # integer range [-qmax, qmax], zero-point 0
        qmax = 2 ** (num_bits - 1) - 1                      # 127 for 8-bit
        max_abs = w.abs().amax(dim=reduce_dims, keepdim=True)
        scale = max_abs / qmax
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))  # avoid /0 (e.g. all-zeros)
        zero_point = torch.zeros_like(scale)
        q = torch.clamp(torch.round(w / scale), -qmax, qmax)
        w_hat = q * scale
    else:
        # affine: map [min, max] -> [0, 2**num_bits - 1] with an integer zero-point
        qmin, qmax = 0, 2 ** num_bits - 1                   # [0, 255] for 8-bit
        w_min = w.amin(dim=reduce_dims, keepdim=True)
        w_max = w.amax(dim=reduce_dims, keepdim=True)
        scale = (w_max - w_min) / (qmax - qmin)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))  # avoid /0 (constant tensor)
        zero_point = torch.round(qmin - w_min / scale)
        q = torch.clamp(torch.round(w / scale + zero_point), qmin, qmax)
        w_hat = (q - zero_point) * scale

    return {"q": q, "scale": scale, "zero_point": zero_point, "w_hat": w_hat}

```

## Part 1b — `quantization_error`


```python
def quantization_error(w, w_hat):
    """Compare a tensor `w` with its reconstruction `w_hat`.

    Return {"mse": ..., "max_abs_err": ..., "snr_db": ...}, where
    snr_db = 10 * log10(mean(w**2) / mse) (use inf when mse == 0).
    """
    w = w.float()
    w_hat = w_hat.float()
    diff = w - w_hat
    mse = torch.mean(diff ** 2).item()
    max_abs_err = diff.abs().max().item()
    signal = torch.mean(w ** 2).item()
    snr_db = float("inf") if mse == 0 else 10.0 * math.log10(signal / mse)
    return {"mse": mse, "max_abs_err": max_abs_err, "snr_db": snr_db}

```

## Self-check (fast, runs anywhere)


```python
# ✅ SELF-CHECK — DO NOT EDIT.
_w = make_weight_matrix()

# --- quantize_dequantize basic contract (8-bit per-tensor symmetric) ---
r8 = quantize_dequantize(_w, 8, symmetric=True, per_channel_dim=None)
assert set(r8) >= {"q", "scale", "zero_point", "w_hat"}, "return dict must have q/scale/zero_point/w_hat"
assert r8["w_hat"].shape == _w.shape, "w_hat must match w's shape"
assert r8["q"].abs().max() <= 127 + 1e-4, "8-bit symmetric codes must lie in [-127, 127]"
e8 = quantization_error(_w, r8["w_hat"])
assert e8["snr_db"] > 12, f"8-bit reconstruction too poor (snr={e8['snr_db']:.1f} dB)"

# symmetric quantization maps exact zero to (near) zero
_z = torch.zeros(4, 4)
assert quantize_dequantize(_z, 8, symmetric=True)["w_hat"].abs().max() < 1e-6

# --- fewer bits => more error ---
r4 = quantize_dequantize(_w, 4, symmetric=True, per_channel_dim=None)
e4 = quantization_error(_w, r4["w_hat"])
assert e4["mse"] > e8["mse"], "4-bit should have larger MSE than 8-bit"

# --- per-channel beats per-tensor on the outlier weight (same bits) ---
r8c = quantize_dequantize(_w, 8, symmetric=True, per_channel_dim=1)
e8c = quantization_error(_w, r8c["w_hat"])
assert e8c["snr_db"] > e8["snr_db"], "per-channel should beat per-tensor here (outlier columns)"

# --- asymmetric helps a shifted, all-positive tensor vs symmetric ---
_pos = _w.abs() + 0.5
sym = quantization_error(_pos, quantize_dequantize(_pos, 6, symmetric=True)["w_hat"])
asym = quantization_error(_pos, quantize_dequantize(_pos, 6, symmetric=False)["w_hat"])
assert asym["mse"] <= sym["mse"] + 1e-12, "asymmetric should not be worse on a shifted tensor"

print("quantize_dequantize / error all consistent   PASS")
print(f"  8-bit  per-tensor : snr={e8['snr_db']:5.1f} dB   mse={e8['mse']:.2e}")
print(f"  8-bit  per-channel: snr={e8c['snr_db']:5.1f} dB   mse={e8c['mse']:.2e}   <-- outliers isolated")
print(f"  4-bit  per-tensor : snr={e4['snr_db']:5.1f} dB   mse={e4['mse']:.2e}")
print()
print("All checks passed")
```

    quantize_dequantize / error all consistent   PASS
      8-bit  per-tensor : snr= 18.1 dB   mse=2.16e-05
      8-bit  per-channel: snr= 42.6 dB   mse=7.57e-08   <-- outliers isolated
      4-bit  per-tensor : snr=  5.1 dB   mse=4.26e-04
    
    All checks passed


## The Q1 official run — accuracy vs bits, and the weight-memory table


```python
# 🔒 DO NOT EDIT — Q1 official run: sweep + plot + 7B weight-memory table.
w = make_weight_matrix()

bit_widths = [8, 6, 4, 3]
schemes = [("per-tensor",  None), ("per-channel", 1)]
rows = []
snr_curves = {name: [] for name, _ in schemes}
for name, dim in schemes:
    for b in bit_widths:
        res = quantize_dequantize(w, b, symmetric=True, per_channel_dim=dim)
        err = quantization_error(w, res["w_hat"])
        snr_curves[name].append(err["snr_db"])
        rows.append({"scheme": name, "bits": b,
                     "snr_db": round(err["snr_db"], 2),
                     "mse": err["mse"], "max_abs_err": err["max_abs_err"]})

print("Weight-quantization accuracy (256x1024 weights with 4 outlier columns)")
print(f"{'scheme':12} {'bits':>4} {'SNR(dB)':>9} {'MSE':>12}")
for r in rows:
    print(f"{r['scheme']:12} {r['bits']:>4} {r['snr_db']:>9.2f} {r['mse']:>12.2e}")

plt.figure(figsize=(6, 4))
for name, _ in schemes:
    plt.plot(bit_widths, snr_curves[name], marker="o", label=name)
plt.gca().invert_xaxis()
plt.xlabel("bits per weight"); plt.ylabel("reconstruction SNR (dB)")
plt.title("Quantization accuracy vs bit-width"); plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
_png = os.path.join(Q1_DIR, "snr_vs_bits.png")
plt.savefig(_png, dpi=120); plt.close()

# --- the payoff: weight memory for a 7B model ---
N = 7_000_000_000
print(f"\n7B model — weight memory by format")
print(f"{'format':12} {'GiB':>8} {'vs fp16':>9}")
base = None
mem_table = []
for fmt, bits in [("fp16/bf16", 16), ("fp8", 8), ("int4", 4)]:
    gib = bytes_for(N, bits) / (1024**3)
    base = base or gib
    mem_table.append({"format": fmt, "bits": bits, "weight_gib": round(gib, 2),
                      "shrink_vs_fp16": round(base / gib, 2)})
    print(f"{fmt:12} {gib:>8.2f} {base/gib:>8.2f}x")

with open(os.path.join(Q1_DIR, "q1_results.json"), "w") as f:
    json.dump({"accuracy": rows, "memory_7b": mem_table}, f, indent=2)
print(f"\nwrote {_png} and results/q1/q1_results.json")
```

    Weight-quantization accuracy (256x1024 weights with 4 outlier columns)
    scheme       bits   SNR(dB)          MSE
    per-tensor      8     18.09     2.16e-05
    per-tensor      6      6.80     2.91e-04
    per-tensor      4      5.15     4.26e-04
    per-tensor      3      4.01     5.53e-04
    per-channel     8     42.65     7.57e-08
    per-channel     6     30.60     1.21e-06
    per-channel     4     17.71     2.36e-05
    per-channel     3     10.35     1.29e-04
    
    7B model — weight memory by format
    format            GiB   vs fp16
    fp16/bf16       13.04     1.00x
    fp8              6.52     2.00x
    int4             3.26     4.00x
    
    wrote results/q1/snr_vs_bits.png and results/q1/q1_results.json


# Question 2 — Serve BF16 vs on-the-fly 8-bit with vLLM + guidellm (H100 **or** L40S)

Now measure the win on real hardware. You will serve the **same model twice** — once in **BF16**, once
**on-the-fly 8-bit** — and benchmark both with **guidellm** against the OpenAI-compatible endpoint. Nothing
is pre-quantized and no checkpoint is swapped: vLLM quantizes the BF16 weights at load time. The only thing
that changes with your GPU is the `--quantization` **method** (the harness picks it as `QUANT_METHOD`):

| Your GPU | `--quantization` | What it does |
|----------|------------------|--------------|
| **H100** (compute cap ≥ 9.0) | `fp8` | per-tensor FP8 weights, on native FP8 tensor cores |
| **L40S** / other (no FP8 datapath) | `int8_per_channel_weight_only` | per-channel INT8 weights (CUTLASS int8 kernels) |

**Model — you pick it.** Choose any **ungated, dense** instruct model on the Hugging Face Hub between
**~3B and ~8B** parameters that vLLM can quantize on the fly, and set it in the cell below. The ~8B ceiling
keeps the BF16 copy fitting the *smaller* supported GPU (L40S, 48 GB) with room for the KV cache; the ~3B
floor keeps the latency numbers out of the noise. Steer clear of **gated** repos (Llama/Gemma need HF auth,
which will stall the benchmark) and **mixture-of-experts / exotic architectures** (the on-the-fly quant
kernels may not cover them). The same id is served both times — only the `--quantization` flag differs.

> **Smoke-test your pick first** (~30 s, saves a failed ~15-min run):
> `vllm serve <model-id> --quantization fp8 --max-model-len 4096` (swap `int8_per_channel_weight_only` on an
> L40S). Once it prints `server is up`, Ctrl-C — your pick is good for the whole assignment.

Both paths cut weight memory ~2× vs BF16 and should lift throughput / cut latency, since fewer bits per weight
means less HBM traffic per token. FP8 on the H100 rides dedicated FP8 tensor cores; the L40S int8 path uses
CUTLASS int8 kernels. Report what you actually measure — the exact latency picture is kernel- and
hardware-dependent.

> **vLLM version:** both `fp8` and `int8_per_channel_weight_only` are registered quantization methods in the
> vLLM pinned by `requirements.txt` (latest). `vllm serve --help | grep -i quantization` lists what your build
> supports if you ever need to check.

## What you implement

| Part | Function | Idea |
|------|----------|------|
| 2a | `vllm_launch_args` | the `vllm serve` CLI args; add `--quantization QUANT_METHOD` when `quantize=True` |
| 2b | `guidellm_command`  | the `guidellm benchmark` argv against the endpoint |
| 2c | `extract_metrics`   | pull TTFT / ITL / throughput out of guidellm's `benchmarks.json` |

The harness owns the fragile plumbing (picking `QUANT_METHOD` for your GPU, starting the server, polling
`/health`, killing it, reading GPU memory). The self-checks run on CPU against a bundled sample report; **the
official run needs a GPU** and takes ~10–15 min (two model loads + four short benchmarks).

## Pick your model (✏️ set this)


```python
# ✏️ YOUR CHOICE — set the model you'll serve (see the guidelines above).
# The SAME id is served twice: once BF16, once on-the-fly 8-bit.
# Requirements: ungated, dense, ~3B–8B params, quantizable on the fly by vLLM.
# Smoke-test it once from a shell before the official run (see the note above).
MODEL = "Qwen/Qwen2.5-7B-Instruct"    # dense, ungated, ~7.6B; fits H100 and L40S
assert MODEL, "Set MODEL to your chosen Hugging Face model id (see the guidelines above)."
print(f"serving: {MODEL}")

```

    serving: Qwen/Qwen2.5-7B-Instruct


## The Q2 harness (DO NOT EDIT)


```python
# 🔒 DO NOT EDIT — Q2 harness: subprocess plumbing + a sample guidellm report.
import os, sys, re, json, time, signal, shutil, subprocess, urllib.request

Q2_DIR = os.path.join("results", "q2")
os.makedirs(Q2_DIR, exist_ok=True)

# MODEL is set by YOU in the selection cell above (same base id is served twice:
# once BF16, once on-the-fly 8-bit).
PORT = 8000
BASE_URL = f"http://localhost:{PORT}"

# Benchmark workload (synthetic prompts): 512 in, 256 out — a typical chat-ish shape.
PROMPT_TOKENS = 512
OUTPUT_TOKENS = 256
MAX_SECONDS = 30


def gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0)
    except Exception:
        return "no-cuda"


def fp8_capable() -> bool:
    """True only on GPUs with native FP8 tensor cores — compute capability >= 9.0
    (H100 / H200 / GH200). L40S and other Ada cards (8.9) and everything older return
    False, so the quantized config uses on-the-fly INT8 instead of FP8."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        return torch.cuda.get_device_capability(0)[0] >= 9
    except Exception:
        return False


# The quantized config is ALWAYS on-the-fly on the SAME base model — only the vLLM
# --quantization method changes with the GPU:
#   * FP8-capable (H100)   -> "fp8"                          (per-tensor FP8 weights)
#   * otherwise (L40S ...) -> "int8_per_channel_weight_only" (per-channel INT8 weights)
# Both quantize the BF16 weights at load time with no calibration and no checkpoint swap.
QUANT_METHOD = "fp8" if fp8_capable() else "int8_per_channel_weight_only"
QUANT_TAG = "fp8" if fp8_capable() else "int8"


def nvidia_smi_used_mib() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True)
        return float(out.strip().splitlines()[0])
    except Exception:
        return float("nan")


def start_server(serve_args: list, log_path: str) -> subprocess.Popen:
    """Launch `vllm serve <serve_args...>`, streaming logs to log_path."""
    cmd = ["vllm", "serve", *serve_args]
    print("  $ " + " ".join(cmd))
    logf = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)


def wait_for_server(proc: subprocess.Popen, timeout: float = 900.0) -> None:
    """Block until GET /health returns 200, or raise (also raises if the process dies)."""
    deadline = time.time() + timeout
    url = f"{BASE_URL}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm exited early (code {proc.returncode}) — check the log")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    print("  server is up")
                    return
        except Exception:
            time.sleep(3)
    raise TimeoutError("vllm did not become healthy in time")


def stop_server(proc: subprocess.Popen) -> None:
    if proc is None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=60)
    except Exception:
        proc.kill(); proc.wait()
    time.sleep(5)  # let the GPU memory actually free


def weights_gib_from_log(log_path: str) -> float:
    """Parse vLLM's 'model weights took X GiB' line — the honest weight-memory number.
    (nvidia-smi is confounded by vLLM pre-allocating the KV cache to fill the GPU.)"""
    try:
        txt = open(log_path).read()
    except Exception:
        return float("nan")
    m = re.search(r"weights? (?:took|take)\s+([\d.]+)\s*GiB", txt, re.IGNORECASE)
    return float(m.group(1)) if m else float("nan")


def run_subprocess(cmd: list, timeout: float) -> int:
    print("  $ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, timeout=timeout).returncode


def _sample_stat(mean, median):
    return {"mean": mean, "median": median,
            "std_dev": 0.0, "count": 100,
            "percentiles": {"p50": median, "p90": median, "p99": median, "p999": median}}


def write_sample_report(path: str) -> None:
    """A schema-faithful guidellm v0.4 report so the self-check runs without a GPU."""
    report = {"args": {"model": "sample"}, "benchmarks": [
        {"metrics": {
            "requests_per_second":      {"successful": _sample_stat(5.0, 5.0)},
            "request_latency":          {"successful": _sample_stat(0.42, 0.40)},
            "time_to_first_token_ms":   {"successful": _sample_stat(45.0, 40.0)},
            "inter_token_latency_ms":   {"successful": _sample_stat(12.0, 11.5)},
            "output_tokens_per_second": {"successful": _sample_stat(850.0, 840.0)},
        }}]}
    with open(path, "w") as f:
        json.dump(report, f)


print(f"Q2 harness ready. gpu={gpu_available()} ({gpu_name()})  fp8_capable={fp8_capable()}")
print(f"  model={MODEL}")
print(f"  quantized config -> {QUANT_TAG.upper()} via --quantization {QUANT_METHOD}")
if not shutil.which("vllm"):
    print("NOTE: `vllm` not on PATH — fine for the CPU self-check; install it for the official run.")
```

    Q2 harness ready. gpu=True (NVIDIA H100 80GB HBM3)  fp8_capable=True
      model=Qwen/Qwen2.5-7B-Instruct
      quantized config -> FP8 via --quantization fp8


## Part 2a — `vllm_launch_args`

Return the argument list that follows `vllm serve` (the harness prepends `vllm serve`). The first element
should be the model id (`model`).

Enable **on-the-fly quantization** with `--quantization <method>` **only when `quantize=True`**. The method is
GPU-dependent and the harness has already picked it for you as **`QUANT_METHOD`**:

- **H100** (`fp8_capable()` is `True`) → `QUANT_METHOD == "fp8"` — per-tensor FP8 weights.
- **L40S / non-FP8 GPU** → `QUANT_METHOD == "int8_per_channel_weight_only"` — per-channel INT8 weights.

Either way it's the same base model, quantized at load time with no calibration — just pass
`--quantization QUANT_METHOD`. That one flag is what turns the BF16 model into an 8-bit one, same weights, half
the bits.

Also set `--port`, a `--max-model-len` (so both configs use the same context and the comparison is fair),
and `--gpu-memory-utilization` (leave headroom; the default fills the GPU). Current vLLM does not log requests
by default, so logs are already quiet — no extra flag needed (older docs used `--disable-log-requests`, which
newer vLLM removed).

Reference: `vllm serve --help`, and the vLLM repo at https://github.com/vllm-project/vllm.


```python
def vllm_launch_args(model, quantize=False, port=PORT, max_model_len=4096):
    """Return the argument list that follows `vllm serve` (see Part 2a above).

    The first element is the model id. Add `--quantization QUANT_METHOD` only when
    quantize=True. Also set --port, --max-model-len, and --gpu-memory-utilization.
    """
    args = [
        model,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", "0.90",   # leave headroom for the KV cache
    ]
    if quantize:
        # QUANT_METHOD is chosen by the harness for this GPU:
        #   "fp8" on an H100, "int8_per_channel_weight_only" otherwise.
        args += ["--quantization", QUANT_METHOD]
    return args

```

## Part 2b — `guidellm_command`

Return the full argv list that benchmarks the running server with guidellm. This build is guidellm 0.7.x,
whose CLI is spec-based (arguments look like `--flag kind=...,key=value`) rather than the old flat flags. The
entry point is `guidellm run`; the bare `guidellm benchmark` now only reloads saved reports.

Work out the exact flag names and spec syntax yourself from the CLI help and the project docs:

- `guidellm run --help`
- guidellm on GitHub: https://github.com/vllm-project/guidellm

Your command needs to cover five things:

1. Where to send requests: an OpenAI-HTTP backend aimed at the served `target` (the base URL) and `model`.
   guidellm reads the tokenizer from `model` by default, so a real HF id needs nothing extra.
2. What to send: a synthetic-text workload built from the given `prompt_tokens` and `output_tokens`.
3. How to drive it: the traffic profile. Use the synchronous profile for clean single-stream latency
   (TTFT and ITL), and the throughput profile for peak tokens per second. The throughput profile requires a
   concurrency, so pass `max_concurrency` only in that case.
4. When to stop: bound each run with a maximum duration in seconds.
5. Where to write it: emit a JSON report to `out_path`, which is the file 2c parses.

Also turn off the interactive console output so the run stays quiet under a subprocess.


```python
def guidellm_command(target, model, out_path, rate_type="synchronous",
                     prompt_tokens=PROMPT_TOKENS, output_tokens=OUTPUT_TOKENS,
                     max_seconds=MAX_SECONDS, max_concurrency=64):
    """Return the argv list for `guidellm run` (see Part 2b above).

    Cover the five things: the OpenAI-HTTP backend (target + model), the synthetic-text
    workload (prompt_tokens / output_tokens), the profile (rate_type; the throughput
    profile needs max_concurrency), a max-duration constraint in seconds, and a JSON
    report written to out_path. Keep the console non-interactive.
    """
    # guidellm 0.7.x is spec-based: each flag takes `kind=...,key=value`.
    if rate_type == "throughput":
        profile = f"kind=throughput,max_concurrency={max_concurrency}"   # saturation run
    else:
        profile = f"kind={rate_type}"                                    # e.g. synchronous latency run
    # `--output` is repeatable, so emit the machine-readable JSON (parsed by 2c) AND the
    # interactive HTML report next to it — the assignment asks for both per run.
    html_path = os.path.splitext(out_path)[0] + ".html"
    return [
        "guidellm", "run",
        # 1) where: OpenAI-HTTP backend aimed at the served target + model
        "--backend", f"kind=openai_http,target={target},model={model}",
        # 2) what: synthetic text workload
        "--data", f"kind=synthetic_text,prompt_tokens={prompt_tokens},output_tokens={output_tokens}",
        # 3) how: traffic profile
        "--profile", profile,
        # 4) when to stop: max duration in seconds
        "--constraint", f"kind=max_duration,seconds={max_seconds}",
        # 5) where to write: JSON report parsed by 2c, plus the HTML report for submission
        "--output", f"kind=json,path={out_path}",
        "--output", f"kind=html,path={html_path}",
        # keep it quiet under a subprocess
        "--disable-console",
    ]
```

## Part 2c — `extract_metrics`

guidellm writes a `benchmarks.json` whose shape is
`report["benchmarks"][i]["metrics"][<name>]["successful"][<stat>]`, where `<name>` is e.g.
`time_to_first_token_ms`, `inter_token_latency_ms`, `output_tokens_per_second`, `request_latency`,
`requests_per_second`, and `<stat>` is `"mean"` / `"median"` (there is also a `percentiles` dict).

Return **a list with one dict per benchmark** in the report, each with keys:
`ttft_ms_median`, `itl_ms_median`, `output_toks_per_s_mean`, `req_latency_s_median`, `rps_mean`.


```python
def extract_metrics(report_path):
    """Parse a guidellm report and return a list with one dict per benchmark.

    Shape: report["benchmarks"][i]["metrics"][<name>]["successful"][<stat>].
    Each dict has keys: ttft_ms_median, itl_ms_median, output_toks_per_s_mean,
    req_latency_s_median, rps_mean (see Part 2c above).
    """
    with open(report_path) as f:
        report = json.load(f)

    rows = []
    for bm in report["benchmarks"]:
        m = bm["metrics"]

        def stat(name, s):
            return m[name]["successful"][s]

        rows.append({
            "ttft_ms_median":         stat("time_to_first_token_ms", "median"),
            "itl_ms_median":          stat("inter_token_latency_ms", "median"),
            "output_toks_per_s_mean": stat("output_tokens_per_second", "mean"),
            "req_latency_s_median":   stat("request_latency", "median"),
            "rps_mean":               stat("requests_per_second", "mean"),
        })
    return rows

```

## Self-check (runs anywhere — uses a bundled sample report)


```python
# ✅ SELF-CHECK — DO NOT EDIT.
# 2a — on-the-fly quantization flag appears only when quantize=True, using the
# GPU-appropriate method the harness chose (QUANT_METHOD: "fp8" on H100, else
# "int8_per_channel_weight_only"). Both configs serve the same base model id.
a_bf16 = vllm_launch_args(MODEL, quantize=False)
a_quant = vllm_launch_args(MODEL, quantize=True)
assert a_bf16[0] == MODEL, "first arg should be the model id"
assert a_quant[0] == MODEL, "first arg should be the model id"
_j_bf16, _j_quant = " ".join(a_bf16), " ".join(a_quant)
assert "--quantization" not in a_bf16, "BF16 config must NOT request quantization"
assert "--quantization" in a_quant, "quantized config must pass --quantization"
assert QUANT_METHOD in _j_quant, f"quantized config must request --quantization {QUANT_METHOD}"
assert "--port" in a_bf16 and "--max-model-len" in a_bf16, "set --port and --max-model-len for a fair run"
print(f"  quantized config on this machine: {QUANT_TAG.upper()}  (--quantization {QUANT_METHOD})")

# 2b — guidellm 0.7.x `run` command has the required specs and target.
cmd = guidellm_command(BASE_URL, MODEL, os.path.join(Q2_DIR, "x.json"),
                       rate_type="synchronous")
assert cmd[0] == "guidellm" and "run" in cmd, "must invoke `guidellm run`"
cj = " ".join(str(c) for c in cmd)
assert BASE_URL in cj and MODEL in cj, "must point the backend at the served target+model"
assert "--data" in cmd and "prompt_tokens" in cj and "output_tokens" in cj, "must pass synthetic --data"
assert "--profile" in cmd and "synchronous" in cj, "must set the profile (rate type)"
assert "seconds" in cj, "must bound each run (constraint max_duration seconds)"
assert any("x.json" in str(c) for c in cmd), "must write the JSON to out_path"
# throughput profile must carry a concurrency
tcj = " ".join(str(c) for c in guidellm_command(BASE_URL, MODEL, os.path.join(Q2_DIR, "x.json"), rate_type="throughput"))
assert "throughput" in tcj and "max_concurrency" in tcj, "throughput profile must pass max_concurrency"

# 2c — parse the bundled sample report.
_sample = os.path.join(Q2_DIR, "_sample_report.json")
write_sample_report(_sample)
m = extract_metrics(_sample)
assert isinstance(m, list) and len(m) == 1, "extract_metrics returns one dict per benchmark"
row = m[0]
assert abs(row["ttft_ms_median"] - 40.0) < 1e-6, "TTFT median should read 40.0 ms from the sample"
assert abs(row["itl_ms_median"] - 11.5) < 1e-6, "ITL median should read 11.5 ms"
assert abs(row["output_toks_per_s_mean"] - 850.0) < 1e-6, "throughput mean should read 850.0 tok/s"
assert abs(row["req_latency_s_median"] - 0.40) < 1e-6, "request-latency median should read 0.40 s"

print("vllm_launch_args / guidellm_command / extract_metrics all OK   PASS")
print()
print("All checks passed")

```

      quantized config on this machine: FP8  (--quantization fp8)
    vllm_launch_args / guidellm_command / extract_metrics all OK   PASS
    
    All checks passed


## The Q2 official run — BF16 vs FP8 on your H100


```python
# 🔒 DO NOT EDIT — Q2 official run. Serves the model twice and benchmarks each.
if not gpu_available():
    print("SKIP: no GPU detected. Q2's reported numbers MUST come from your GPU run.")
else:
    print(f"GPU: {gpu_name()}   quantized config = {QUANT_TAG.upper()} (--quantization {QUANT_METHOD})")
    QLABEL = QUANT_TAG.upper()
    CONFIGS = [("bf16", False), (QUANT_TAG, True)]
    PROFILES = ["synchronous", "throughput"]   # latency run, then saturation run
    summary = {}

    for tag, quantize in CONFIGS:
        print(f"\n=== {tag.upper()} =====================================")
        log_path = os.path.join(Q2_DIR, f"vllm_{tag}.log")
        proc = None
        try:
            proc = start_server(vllm_launch_args(MODEL, quantize=quantize), log_path)
            wait_for_server(proc)
            time.sleep(3)
            w_gib = weights_gib_from_log(log_path)
            smi = nvidia_smi_used_mib()
            print(f"  weights ~= {w_gib:.2f} GiB   (nvidia-smi used: {smi:.0f} MiB, incl. KV cache)")

            per_profile = {}
            for prof in PROFILES:
                out_json = os.path.join(Q2_DIR, f"bench_{tag}_{prof}.json")
                rc = run_subprocess(
                    guidellm_command(BASE_URL, MODEL, out_json, rate_type=prof),
                    timeout=900)
                if rc != 0:
                    print(f"  guidellm ({prof}) exited {rc}"); continue
                rows = extract_metrics(out_json)
                # take the most-loaded benchmark for throughput, first for latency
                row = max(rows, key=lambda r: r["output_toks_per_s_mean"]) if prof == "throughput" else rows[0]
                per_profile[prof] = row
                print(f"  [{prof:11}] TTFT={row['ttft_ms_median']:.1f}ms  "
                      f"ITL={row['itl_ms_median']:.2f}ms  "
                      f"out={row['output_toks_per_s_mean']:.0f}tok/s")
            summary[tag] = {"weights_gib": w_gib, "profiles": per_profile}
        finally:
            stop_server(proc)

    # ---- comparison ----
    if "bf16" in summary and QUANT_TAG in summary:
        b, f = summary["bf16"], summary[QUANT_TAG]
        def g(cfg, prof, key):
            return cfg["profiles"].get(prof, {}).get(key, float("nan"))
        mem_ratio = b["weights_gib"] / f["weights_gib"] if f["weights_gib"] else float("nan")
        tput_ratio = (g(f, "throughput", "output_toks_per_s_mean")
                      / g(b, "throughput", "output_toks_per_s_mean"))
        print(f"\n================ BF16 vs {QLABEL} ================")
        print(f"{'metric':28}{'BF16':>12}{QLABEL:>12}{QLABEL + ' / BF16':>14}")
        print(f"{'weight memory (GiB)':28}{b['weights_gib']:>12.2f}{f['weights_gib']:>12.2f}{1/mem_ratio:>13.2f}x")
        print(f"{'peak throughput (tok/s)':28}"
              f"{g(b,'throughput','output_toks_per_s_mean'):>12.0f}"
              f"{g(f,'throughput','output_toks_per_s_mean'):>12.0f}{tput_ratio:>13.2f}x")
        print(f"{'TTFT median (ms)':28}"
              f"{g(b,'synchronous','ttft_ms_median'):>12.1f}"
              f"{g(f,'synchronous','ttft_ms_median'):>12.1f}"
              f"{g(f,'synchronous','ttft_ms_median')/g(b,'synchronous','ttft_ms_median'):>13.2f}x")
        print(f"{'ITL median (ms)':28}"
              f"{g(b,'synchronous','itl_ms_median'):>12.2f}"
              f"{g(f,'synchronous','itl_ms_median'):>12.2f}"
              f"{g(f,'synchronous','itl_ms_median')/g(b,'synchronous','itl_ms_median'):>13.2f}x")

        with open(os.path.join(Q2_DIR, "comparison.json"), "w") as fp:
            json.dump(summary, fp, indent=2)

        labels = ["weight GiB", "peak tok/s", "TTFT ms", "ITL ms"]
        bf = [b["weights_gib"], g(b,"throughput","output_toks_per_s_mean"),
              g(b,"synchronous","ttft_ms_median"), g(b,"synchronous","itl_ms_median")]
        fp = [f["weights_gib"], g(f,"throughput","output_toks_per_s_mean"),
              g(f,"synchronous","ttft_ms_median"), g(f,"synchronous","itl_ms_median")]
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
        for ax, lab, vb, vf in zip(axes, labels, bf, fp):
            ax.bar(["BF16", QLABEL], [vb, vf], color=["#4C72B0", "#DD8452"])
            ax.set_title(lab); ax.grid(True, axis="y", alpha=0.3)
        fig.suptitle(f"{MODEL} — BF16 vs on-the-fly {QLABEL} on {gpu_name()}")
        fig.tight_layout()
        fig.savefig(os.path.join(Q2_DIR, "bf16_vs_fp8.png"), dpi=120); plt.close(fig)
        print("\nwrote results/q2/comparison.json, bf16_vs_fp8.png, and per-run guidellm .json/.html")
        print("open the guidellm .html reports in a browser for the full latency/throughput curves")

```

    GPU: NVIDIA H100 80GB HBM3   quantized config = FP8 (--quantization fp8)
    
    === BF16 =====================================
      $ vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.90


      server is up


      weights ~= nan GiB   (nvidia-smi used: 76521 MiB, incl. KV cache)
      $ guidellm run --backend kind=openai_http,target=http://localhost:8000,model=Qwen/Qwen2.5-7B-Instruct --data kind=synthetic_text,prompt_tokens=512,output_tokens=256 --profile kind=synchronous --constraint kind=max_duration,seconds=30 --output kind=json,path=results/q2/bench_bf16_synchronous.json --output kind=html,path=results/q2/bench_bf16_synchronous.html --disable-console


    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.


      [synchronous] TTFT=22.1ms  ITL=5.95ms  out=166tok/s
      $ guidellm run --backend kind=openai_http,target=http://localhost:8000,model=Qwen/Qwen2.5-7B-Instruct --data kind=synthetic_text,prompt_tokens=512,output_tokens=256 --profile kind=throughput,max_concurrency=64 --constraint kind=max_duration,seconds=30 --output kind=json,path=results/q2/bench_bf16_throughput.json --output kind=html,path=results/q2/bench_bf16_throughput.html --disable-console


      [throughput ] TTFT=378.7ms  ITL=8.54ms  out=5859tok/s


    
    === FP8 =====================================
      $ vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.90 --quantization fp8


      server is up


      weights ~= nan GiB   (nvidia-smi used: 76509 MiB, incl. KV cache)
      $ guidellm run --backend kind=openai_http,target=http://localhost:8000,model=Qwen/Qwen2.5-7B-Instruct --data kind=synthetic_text,prompt_tokens=512,output_tokens=256 --profile kind=synchronous --constraint kind=max_duration,seconds=30 --output kind=json,path=results/q2/bench_fp8_synchronous.json --output kind=html,path=results/q2/bench_fp8_synchronous.html --disable-console


      [synchronous] TTFT=24.5ms  ITL=4.12ms  out=238tok/s
      $ guidellm run --backend kind=openai_http,target=http://localhost:8000,model=Qwen/Qwen2.5-7B-Instruct --data kind=synthetic_text,prompt_tokens=512,output_tokens=256 --profile kind=throughput,max_concurrency=64 --constraint kind=max_duration,seconds=30 --output kind=json,path=results/q2/bench_fp8_throughput.json --output kind=html,path=results/q2/bench_fp8_throughput.html --disable-console


    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.


      [throughput ] TTFT=308.0ms  ITL=6.30ms  out=8070tok/s


    
    ================ BF16 vs FP8 ================
    metric                              BF16         FP8    FP8 / BF16
    weight memory (GiB)                  nan         nan          nanx
    peak throughput (tok/s)             5859        8070         1.38x
    TTFT median (ms)                    22.1        24.5         1.11x
    ITL median (ms)                     5.95        4.12         0.69x
    
    wrote results/q2/comparison.json, bf16_vs_fp8.png, and per-run guidellm .json/.html
    open the guidellm .html reports in a browser for the full latency/throughput curves


## Q2 analysis — what changed, and why

You've served the model twice and benchmarked each — now **read** the numbers instead of just producing them.
The official run above printed medians and the BF16-vs-quantized ratios. The cell below reloads your saved
guidellm reports (`results/q2/bench_*.json`) and adds the **tail latencies** (p90 / p99), which frequently move
differently from the median under load. Run it, eyeball the ratios, then answer the writeup questions.


```python
# 🔒 DO NOT EDIT — Q2 analysis: reload the saved guidellm reports and lay out the
# BF16-vs-quantized differences, INCLUDING tail latencies (p90/p99), for the writeup below.
import os, json

def _pick(rep, prof):
    """The benchmark to read from a report: the most-loaded one for throughput, else the first."""
    bms = rep["benchmarks"]
    if prof == "throughput":
        return max(bms, key=lambda b: b["metrics"]["output_tokens_per_second"]["successful"]["mean"])["metrics"]
    return bms[0]["metrics"]

def _load(tag, prof):
    p = os.path.join(Q2_DIR, f"bench_{tag}_{prof}.json")
    return _pick(json.load(open(p)), prof) if os.path.exists(p) else None

def _mid(metrics, name, stat):            # top-level median / mean
    return metrics[name]["successful"][stat] if metrics else float("nan")

def _pct(metrics, name, p):               # tail percentile, tolerant of key spelling
    if not metrics:
        return float("nan")
    pcts = metrics[name]["successful"].get("percentiles", {})
    for k in (f"p{p}", str(p), f"{p}.0", f"{float(p)}"):
        if k in pcts:
            return pcts[k]
    return float("nan")                   # this guidellm build labels percentiles differently

QUANT = QUANT_TAG
sync = {t: _load(t, "synchronous") for t in ("bf16", QUANT)}
tput = {t: _load(t, "throughput")  for t in ("bf16", QUANT)}

if sync["bf16"] is None and tput["bf16"] is None:
    print("No Q2 reports in results/q2/ yet — run the official Q2 cell on your GPU first.")
else:
    def ratio(q, b):
        return q / b if b else float("nan")

    print(f"Single-stream latency (synchronous)         BF16    {QUANT.upper():>6}   quant/BF16")
    for label, name, stat, p in [
        ("TTFT median (ms)",       "time_to_first_token_ms", "median", None),
        ("TTFT p90 (ms)",          "time_to_first_token_ms", None,     90),
        ("TTFT p99 (ms)",          "time_to_first_token_ms", None,     99),
        ("ITL  median (ms)",       "inter_token_latency_ms", "median", None),
        ("ITL  p90 (ms)",          "inter_token_latency_ms", None,     90),
        ("ITL  p99 (ms)",          "inter_token_latency_ms", None,     99),
        ("req latency median (s)", "request_latency",        "median", None),
    ]:
        b = _mid(sync["bf16"], name, stat) if stat else _pct(sync["bf16"], name, p)
        q = _mid(sync[QUANT], name, stat) if stat else _pct(sync[QUANT], name, p)
        print(f"  {label:26}{b:>8.2f}{q:>9.2f}{ratio(q,b):>12.2f}x")

    print(f"\nSaturation throughput (throughput)          BF16    {QUANT.upper():>6}   quant/BF16")
    for label, name in [("output tok/s (mean)", "output_tokens_per_second"),
                        ("requests/s (mean)",   "requests_per_second")]:
        b = _mid(tput["bf16"], name, "mean"); q = _mid(tput[QUANT], name, "mean")
        print(f"  {label:26}{b:>8.1f}{q:>9.1f}{ratio(q,b):>12.2f}x")

    cj = os.path.join(Q2_DIR, "comparison.json")
    if os.path.exists(cj):
        s = json.load(open(cj))
        wb = s.get("bf16", {}).get("weights_gib"); wq = s.get(QUANT, {}).get("weights_gib")
        if wb and wq and wb == wb and wq == wq:     # both present and not nan
            print(f"\nweight memory: BF16 {wb:.2f} GiB  ->  {QUANT.upper()} {wq:.2f} GiB   ({ratio(wb,wq):.2f}x smaller)")

    print("\nHow to read this:")
    print("  * latency ratio  < 1.00  => quantized is faster; throughput ratio > 1.00 => more tokens/s.")
    print("  * ITL ratio vs TTFT ratio: which drops more? (decode is memory-bound, prefill compute-bound)")
    print("  * p99 vs median: do the TAILS shrink as much as the middle, or does quant mostly help the median?")

```

    Single-stream latency (synchronous)         BF16       FP8   quant/BF16
      TTFT median (ms)             22.06    24.55        1.11x
      TTFT p90 (ms)                22.88    25.40        1.11x
      TTFT p99 (ms)                49.60    44.68        0.90x
      ITL  median (ms)              5.95     4.12        0.69x
      ITL  p90 (ms)                 5.95     4.12        0.69x
      ITL  p99 (ms)                 5.96     4.12        0.69x
      req latency median (s)        1.54     1.07        0.70x
    
    Saturation throughput (throughput)          BF16       FP8   quant/BF16
      output tok/s (mean)         5859.0   8070.1        1.38x
      requests/s (mean)             23.5     32.0        1.36x


    
    How to read this:
      * latency ratio  < 1.00  => quantized is faster; throughput ratio > 1.00 => more tokens/s.
      * ITL ratio vs TTFT ratio: which drops more? (decode is memory-bound, prefill compute-bound)
      * p99 vs median: do the TAILS shrink as much as the middle, or does quant mostly help the median?


## Q2 writeup — analyze BF16 vs 8-bit

Using **your own measured numbers** from the comparison table and the analysis cell above (H100 → FP8,
L40S → INT8), answer the following. A few sentences each — reason from *your* data, don't just restate theory.

**Q2.1 — Memory & throughput.** How much smaller is the quantized weight memory vs BF16, and how does peak
throughput (tok/s) compare? Q1 predicted 8 bits ≈ half the weight bytes — does peak throughput improve by that
same ~2×? If it's less, explain the gap: what else is a *saturated* server bound by besides weight bandwidth?

**Q2.2 — Prefill vs decode.** Did **TTFT** or **ITL** improve more (compare the two quant/BF16 ratios)? Explain
with the roofline idea: prefill runs one big matmul over the whole prompt, while decode emits one token at a
time. Which phase is compute-bound and which is memory-bound, and why does dropping to 8 bits help the
memory-bound one more? If you ran INT8 on an L40S (no native FP8 tensor cores) rather than FP8 on an H100, how
might this picture differ?

**Q2.3 — Accuracy.** This quantization is applied **on the fly with no calibration**. Where is it most likely
to hurt model quality, and would that degradation show up *anywhere* in your latency/throughput benchmark?
Name a concrete way you would verify the quantized model is still accurate enough before shipping it.

**Q2.4 — Production.** (a) Which **one or two** metrics from your runs — consider medians *and* the p99 tails
from the analysis cell — would you put on a serving dashboard to catch SLO violations and saturation, and why?
(b) Name one alternative serving engine and when you'd reach for it over vLLM.

### Q2 writeup — analyze BF16 vs 8-bit (measured on an H100, BF16 vs FP8)

**Q2.1 —** The weights shrank from **14.29 GiB in BF16 to 8.2 GiB in FP8**, about **1.74× smaller** rather than the clean 2× Q1's back-of-the-envelope math predicts. Where those two numbers come from matters, because the official-run cell above prints `nan` for weight memory: its regex looks for `weights took X GiB`, while vLLM 0.25.1 logs `Model loading took X GiB`. I left the `DO NOT EDIT` cell alone and read the values straight out of the load logs — `results/q2/vllm_bf16.log:32` (`Model loading took 14.29 GiB`) and `results/q2/vllm_fp8.log:90` (`Model loading took 8.2 GiB`).

The 1.74× is exactly what you get once you account for what *doesn't* get quantized. Qwen2.5-7B is 7.62B params, and 7.62B × 2 bytes = 14.19 GiB — which is the `Checkpoint size: 14.19 GiB` line in the same log, so the parameter count checks out rather than being assumed. The token embeddings and the untied LM head are 152064 × 3584 × 2 ≈ 1.09B of those params and stay in BF16 (~2.0 GiB); the remaining ~6.5B drop to one byte each (~6.1 GiB); layernorms and the per-tensor FP8 scale factors add a little on top. That predicts ~8.1 GiB against the 8.2 GiB measured — within about 1%.

Peak throughput went from **5859 tok/s to 8070 tok/s, a 1.38× gain** (requests/s moved the same way, 23.5 → 32.0, 1.36×) — real, but well short of 2×. The main reason is that a *saturated* server isn't bandwidth-bound the way a single stream is: at 64 concurrent requests, decode stops being a stack of matrix-vector products and becomes matrix-matrix. Arithmetic intensity climbs with batch size, so the server has already drifted off the memory-bound part of the roofline, and halving the weight bytes buys back correspondingly less time. On top of that, everything FP8 doesn't touch still costs full price: the KV cache is still 16-bit (`kv_cache_dtype` left at `auto`, so FP8 never went near it), attention compute grows with sequence length, and there's per-step scheduler and kernel-launch overhead. Those together are what cap the win at ~1.4×.

**Q2.2 —** Decode is the clear winner; prefill is a wash. ITL went **5.95 ms → 4.12 ms, a 0.69× ratio (~31% faster)**, and p90 and p99 moved by the *same* factor (5.947 → 4.120 and 5.957 → 4.124). That the whole distribution shifts together — not just the middle of it — is what makes the number trustworthy.

TTFT I'd call **unchanged**, and the reason is sample size rather than physics. Reading `metrics.time_to_first_token_ms.successful` out of `results/q2/bench_bf16_synchronous.json` and `bench_fp8_synchronous.json`: `count` is **20 (BF16) and 28 (FP8)** requests with `std_dev` of **6.13 ms and 3.75 ms**, giving standard errors of 1.37 ms and 0.71 ms. The gap in means — 23.74 vs 25.29 ms — is **1.55 ms, almost exactly one standard error of the difference**. There is no result there. The medians (22.1 → 24.5 ms, 1.11×) look like a regression only because 20 samples of a right-skewed distribution wander, and the p99 moving the *other* way (49.6 → 44.7 ms) says the same thing from the opposite end.

The contrast with ITL is the whole point. A 30-second synchronous run yields only ~20 TTFT measurements — one per request — but 5100 and 7140 inter-token intervals, at 256 output tokens apiece, with a `std_dev` of 0.004 ms. ITL is measured thousands of times per run and reproduces to three decimal places; TTFT is measured twenty times and does not. Re-running this identical configuration bore that out: ITL came back within 0.002 ms of the previous run, while the TTFT median gap moved from 3.3 ms to 2.5 ms. (That earlier run's synchronous reports are kept under `results/q2/run1_archive/` so the comparison is checkable; every number I quote above is from the current run.) Same conclusion, arrived at empirically — the ITL win is real, the TTFT difference is noise.

The roofline explains why decode is where the win shows up at all. Decode emits one token at a time, streaming the *entire* weight matrix through the cores for a batch of tiny matrix-vector products — very low arithmetic intensity, so the GPU spends most of its time waiting on HBM. Cutting the weights to 8 bits roughly halves that traffic, and decode speeds up almost in proportion. Prefill is the opposite: all 512 prompt tokens go through in one big matrix-matrix multiply, with enough arithmetic per byte loaded to keep the tensor cores busy.

FP8 does not, however, leave prefill on a flat roofline on this GPU. H100 FP8 tensor cores roughly double peak dense FLOPs over BF16, so FP8 **raises the roof** rather than pinning prefill to it. Prefill shows no visible gain here for more mundane reasons: at 512 tokens the prefill GEMMs are small enough that dynamic per-token activation quantization and kernel-launch overhead eat whatever headroom the extra FLOPs offer, and twenty samples can't resolve a couple of milliseconds either way.

On an L40S the picture genuinely would differ, and *there* the flat-roofline argument is the correct one. Ada has no FP8 datapath, so `int8_per_channel_weight_only` is **weight-only**: weights are stored as INT8 and dequantized back to 16-bit for the matmul itself. That still cuts HBM traffic per token, so I'd expect the decode/ITL win to survive in some form — but prefill gets no additional compute throughput whatsoever, so a compute-bound prefill really is pinned to a flat roof there. There'd simply be no mechanism for prefill to improve — which is a weaker and more defensible claim than predicting a measurable TTFT regression, given that this benchmark can't resolve TTFT differences of that size in the first place.

**Q2.3 —** The accuracy risk lives in what a single shared scale can't represent. This FP8 path is applied on the fly with no calibration data and uses one scale per tensor, so a tensor containing a handful of large-magnitude outlier channels has to represent everything else with a step size dictated by its largest value — the failure mode Q1 demonstrated. It's worth being precise about how far that carries over, though: Q1 showed it for **INT8**, where every level is evenly spaced, and FP8 E4M3 carries a per-value exponent, so it tolerates a wide dynamic range far better than a linear 8-bit grid does. The shared per-tensor scale still bites at the extremes, but the outlier story is much more an INT8 one than an FP8 one. The tidy connection is that Q1's *remedy* was per-channel scales — which is exactly what the L40S path (`int8_per_channel_weight_only`) uses and what per-tensor FP8 does not.

However it manifests, it tends to surface as small degradations on the *hardest* inputs — long-context reasoning, arithmetic, code — where a few wrong logits compound, rather than as obvious breakage. The important point is that **none of this would show up anywhere in the latency/throughput benchmark**: guidellm fires synthetic tokens and measures speed, not correctness, so a model that's a couple of points dumber and a model that's perfect look identical on this dashboard. To catch it you have to measure quality directly — run a task eval like MMLU or GSM8K (or a domain-specific set) against both the BF16 and FP8 servers and compare scores, and/or compute perplexity on held-out text — and only ship the FP8 build if the gap is inside an acceptable tolerance. Diffing generations on a fixed prompt set is a cheap sanity check, but a scored eval is what gives you a real go/no-go.

**Q2.4 —** (a) I'd put **p99 ITL** and **saturated throughput** (output tok/s, or requests/s) on the dashboard. p99 ITL rather than the median because the tail is what users feel — a p99 stall in inter-token latency is what makes a stream visibly stutter, and it's the first metric to blow up when the server starts saturating, so it's the earliest honest warning of an SLO violation. It also has the sampling density to be trustworthy in real time, which TTFT demonstrably does not: in this run p99 ITL sat right on top of the median (4.124 vs 4.117 ms) off 7140 samples, whereas TTFT couldn't separate two configurations off 28. Under real load that ITL gap is exactly what opens up first. Throughput is the capacity half of the picture: watching output tok/s against a known ceiling — here ~8070 tok/s on FP8 — tells you how much headroom is left before latency falls off a cliff. If I could only keep one it'd be p99 ITL, since latency SLOs are what users actually notice; throughput is the second because it tells you *why* the latency is moving. (b) I'd reach for **TensorRT-LLM** when I want to squeeze maximum performance out of NVIDIA hardware for a fixed, high-volume model — it compiles a model-specific engine and often beats vLLM on raw latency and throughput, at the cost of a much heavier build step and far less flexibility. vLLM stays my default when I value fast iteration, easy model swapping, and exactly the kind of on-the-fly quantization we just used; I'd only move to TensorRT-LLM once the model and traffic pattern are locked in and every last percent of performance matters. (SGLang is the other one I'd consider, specifically when the workload has lots of shared prefixes or structured outputs, where its prefix caching pays off.)
