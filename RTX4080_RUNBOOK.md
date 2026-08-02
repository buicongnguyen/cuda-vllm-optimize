# RTX 4080 Super reproduction runbook

This is the executable path for reproducing the **experiment method** from the
Viettel AI Race analysis on a Windows PC with an RTX 4080 Super (16 GB). It does
not claim that local latency or ERS equals an H200 MIG result.

## What is verified on this PC

The smoke path was executed end to end on 2026-08-02 with:

| Component | Verified value |
|---|---|
| Host GPU | NVIDIA GeForce RTX 4080 SUPER, 16,376 MiB, compute capability 8.9 |
| Windows driver | 591.86 |
| Runtime | Ubuntu 22.04 on WSL2, kernel 6.18.33.2 |
| Python | 3.12.13 |
| vLLM | 0.25.1 |
| PyTorch / CUDA build | 2.11.0+cu130 / CUDA 13.0 |
| Model | `LiquidAI/LFM2.5-1.2B-Instruct` revision `868df74d...` |
| Smoke result | 4 requested, 4 successful, 0 failed |

The first model start may download weights and compile/cache Ada kernels. A
later start is faster. The smoke test only proves that installation, model
loading, streaming and evidence capture work; its tiny cold-run latency is not
a benchmark score.

## 1. Prerequisites

From PowerShell:

```powershell
wsl --list --verbose
wsl -d Ubuntu-22.04 -- nvidia-smi
```

You need Ubuntu 22.04 running as WSL version 2 and the GPU visible inside it.
Keep the NVIDIA display driver on Windows; do not install a Linux display
driver inside WSL. The setup needs `curl` and `rsync` in Ubuntu. On the tested
machine both were already present. If doctor reports one missing:

```bash
sudo apt-get update
sudo apt-get install -y curl rsync
```

## 2. One-command setup and smoke test

Run this from the repository root in PowerShell:

```powershell
.\scripts\rtx4080_bootstrap.ps1 -Run smoke
```

The command:

1. detects the real RTX 4080 Super through WSL;
2. mirrors the checkout from `/mnt/c` to `~/src/cuda-vllm-optimize` on WSL ext4;
3. installs `uv`, Python 3.12 and a dedicated virtual environment;
4. pins vLLM 0.25.1 and installs the replay harness;
5. supplies a portable C compiler for Triton when Ubuntu has no compiler;
6. starts vLLM, waits for health, sends four streamed requests and stops it;
7. saves the exact config, software/hardware manifest, raw JSONL and server log.

Setup is idempotent. Re-running it reuses the environment and model cache.

Use setup without a run or only run the environment checks:

```powershell
.\scripts\rtx4080_bootstrap.ps1
.\scripts\rtx4080_bootstrap.ps1 -DoctorOnly
```

## 3. Run from inside Ubuntu WSL

After setup:

```bash
source ~/.venvs/lfm-racebench-rtx4080/bin/activate
cd ~/src/cuda-vllm-optimize

python scripts/rtx4080_lab.py doctor
python scripts/rtx4080_lab.py command
python scripts/rtx4080_lab.py run --mode smoke
```

`command` prints the exact `vllm serve` invocation without starting it. The
checked-in baseline is [configs/vllm/rtx4080-r0.args](configs/vllm/rtx4080-r0.args).

## 4. Reproduce the 70 × 6 workload

The baseline run creates 70 conversations with six causal turns each: 420
requests in total. Poisson arrivals use a reproducible seed. The rate of 7
requests/s is an explicit local assumption because the original article did not
publish the contest lambda.

From PowerShell:

```powershell
.\scripts\rtx4080_bootstrap.ps1 -Run baseline
```

Or from Ubuntu WSL:

```bash
python scripts/rtx4080_lab.py run --mode baseline
```

Override workload assumptions without editing code:

```bash
python scripts/rtx4080_lab.py run --mode baseline \
  --conversations 70 --turns 6 --rate 7 \
  --max-tokens 64 --seed 2025
```

Do not compare the local absolute ERS with the contest leaderboard. Use the
local run to validate semantics and compare candidates on the same machine.

## 5. Run a controlled R0/B/R0′ experiment

The first candidate changes exactly one server option: prefix caching. The
orchestrator audits this diff before it starts anything and then runs:

```text
R0 baseline → B prefix-cache candidate → R0′ baseline return → paired comparison
```

```powershell
.\scripts\rtx4080_bootstrap.ps1 -Run aba
```

or:

```bash
python scripts/rtx4080_lab.py run --mode aba
```

The comparison reports paired mean/median/p95/p99 deltas, bootstrap 95%
confidence intervals, faster/slower/uncertain classification, ERS delta and
R0→R0′ drift. A candidate is not promoted when its interval crosses zero or
when baseline-return drift is comparable to the apparent gain.

### Verified A/B/A result on this PC

The complete default block ran in about four minutes and all three stages
finished 420/420 requests with zero failures:

| Stage | Mean TTFT | Mean TPOT | Quoted-formula ERS |
|---|---:|---:|---:|
| R0 | 31.579 ms | 4.138 ms | 65.832 |
| B · prefix cache | 19.729 ms | 3.943 ms | 70.184 |
| R0′ | 17.743 ms | 3.973 ms | 70.460 |

B looked faster than the first R0 and its paired 95% CIs were below zero. It
still **must not be promoted from this block**: unchanged R0′ improved by
4.629 ERS, more than B's 4.352. The initial R0 was affected by warm-up,
clock or persistent cache state. This is the concrete reason the workflow
requires baseline return instead of accepting a simple R0/B comparison. The
machine-readable summary is
[data/rtx4080-verified-aba-summary.json](data/rtx4080-verified-aba-summary.json).

To test another hypothesis, copy the candidate args file, make one change and
pass it explicitly:

```bash
python scripts/rtx4080_lab.py run --mode aba \
  --candidate-config configs/vllm/my-one-change.args
```

## 6. Understand the RTX-specific choices

The local baseline deliberately starts conservatively:

- `--max-model-len=4096`: less state and graph pressure than the article's 8192.
- `--gpu-memory-utilization=0.88`: leaves room on a 16 GB desktop GPU.
- `--max-num-seqs=80`: covers 70 concurrent conversations with margin.
- BF16/`dtype=auto`: avoids mixing quantization into the first baseline.
- pinned model revision and seed: prevents silent model/config changes.
- `VLLM_USE_FLASHINFER_SAMPLER=0`: vLLM 0.25.1 otherwise may JIT a FlashInfer
  sampler for Ada and require a full `nvcc` toolkit. The supported torch sampler
  lets a stock WSL2 machine run reproducibly.
- portable Zig C compiler: Triton needs a host linker for generated launchers;
  the setup provides one when `gcc`/`clang` is absent.

These choices are a safe starting family, not a winning configuration. Change
one variable at a time and create a new baseline family after any vLLM, model,
PyTorch, driver or graph-mode change.

## 7. Evidence produced by every run

Each timestamped folder under `results/rtx4080/` contains:

```text
experiment-plan.json       workload, seed, configs, exact commands and A/B diff
doctor.json                preflight result
R0-*.args                  copied immutable server arguments
R0-*-manifest.json         GPU, driver, packages, CUDA build and source commit
R0-*-server.log            model load, graph capture, warnings and errors
R0-*.jsonl                 one raw record per request plus summary
comparison.json            paired statistics and drift report for A/B/A
```

`results/rtx4080/` is intentionally git-ignored: raw runs can be large and are
machine evidence, not source. Attach a selected result folder to a release or
issue when sharing a conclusion.

## 8. Logic for the next performance step

Use this decision order after a clean A/B/A block:

1. If requests fail or output semantics differ, fix correctness first.
2. If R0′ moved materially from R0, stabilize clocks/temperature/background
   load and repeat; do not credit B.
3. If TTFT improves but TPOT regresses, inspect prefill/decode separately.
4. If both confidence intervals cross zero, repeat enough blocks to estimate
   the noise floor before adding another flag.
5. If a capacity feature helps only near concurrency 70, preserve queue depth,
   cache hit rate and batch-size evidence; an idle microbenchmark cannot explain it.
6. Profile with Nsight Systems only after the harness is stable. Use the timeline
   to select a critical path; use Nsight Compute only on a selected kernel.
7. Implement kernel fusion only if launches remain separate outside graph or
   compiler fusion and their measured contribution can exceed the noise floor.

The deeper terminology and code-learning sequence is on the
[learning page](https://buicongnguyen.github.io/cuda-vllm-optimize/learn.html),
while the visual experiment logic is on the
[decision-flow page](https://buicongnguyen.github.io/cuda-vllm-optimize/decision-flow.html).

## Troubleshooting

**`torch.cuda.is_available()` is false** — update WSL and the Windows NVIDIA
driver, confirm the distro is WSL2, then retry `nvidia-smi` inside Ubuntu.

**Server exits during graph capture** — close other GPU applications. If it is
really VRAM pressure, create a new config with utilization 0.82. Do not silently
mix eager-mode results with graph-mode results.

**`failed to find a C compiler`** — rerun the setup script. It installs the
tested portable compiler wrapper when no system compiler exists.

**FlashInfer asks for `nvcc`** — launch through `rtx4080_lab.py`; it sets the
documented torch-sampler fallback. If you start `vllm serve` manually, export
`VLLM_USE_FLASHINFER_SAMPLER=0` first.

**The Windows checkout and WSL copy differ** — rerun the PowerShell bootstrap.
The Windows checkout is the source and is mirrored to WSL before execution.
