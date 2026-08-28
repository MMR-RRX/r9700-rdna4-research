# Dual AMD R9700 LLM Serving Lab

**Benchmarking and tuning how to run open-weight LLMs — currently `Qwen3.8-27B` — on a dual**
**AMD Radeon AI PRO R9700 (gfx1201 / RDNA 3.5) box, comparing vLLM backends, quantizations, and**
**llama.cpp to find the best setup for each workload.**

> Goal: measure, don't guess. Same config, same inputs, only the thing under test changes.
> Every number in this repo comes from recorded runs (no AI-generated benchmark values).

[![vLLM](https://img.shields.io/badge/vLLM-Apache_2.0-blue)](https://github.com/vllm-project/vllm)
[![RDNA4 port](https://img.shields.io/badge/RDNA4%20port-Apache_2.0-orange)](https://github.com/Capicua25x/vllm-rocm-rdna4)
![ROCm](https://img.shields.io/badge/ROCm-7.2.4-8b5cf6)

---

## TL;DR — the headline result (2× R9700, `Qwen3.8-27B`, TP2)

The best all-round setup on this box is **vLLM RDNA4 port with `TRITON_ATTN` on the FP8
checkpoint**. TRITON beats the AITER attention backend by **+25–40% on long-context decode**;
FP8 beats the MXFP4 export by **+11–15% on decode**; and **AITER is capped at 128K context**
(only TRITON serves the native 262K window). A caveat worth knowing: **AITER *can* load MXFP4**,
but via a slow Triton dequant fallback (~2.6× slower than AITER+FP8).

| setup | decode @64K | decode @128K | decode @256K | concurrency c32 | agent task |
|---|---|---|---|---|---|
| **vLLM TRITON · FP8** | **70.3 t/s** | **66.7 t/s** | **55.4 t/s** | **673 t/s** | **fastest (348 s, 212 LOC)** |
| vLLM TRITON · MXFP4 | 59.5 t/s | 59.1 t/s | 33.0 t/s | 551 t/s | 61 t/s |
| vLLM AITER · FP8 | 56.3 t/s | 47.7 t/s | n/a (128K cap) | 636 t/s | 75 t/s |
| vLLM AITER · MXFP4 | 28.1 t/s | 25.6 t/s | n/a (128K cap) | 359 t/s | 28 t/s (slowest) |

Pick by workload:
- **Burst short-query concurrency** → TRITON·FP8.
- **Max context / long-window fit (256K)** → TRITON (either model, FP8 faster).
- **Single-stream long-context decode** → TRITON·FP8.
- **MXFP4** buys memory/context density, not speed — use it only when you need the room.

Full numbers, tables, and 9 charts (incl. quadrant leaderboards, concurrency curves, pies) are in
**[`benchmark-campaign/report/REPORT.md`](benchmark-campaign/report/REPORT.md)** and
**[`benchmark-campaign/report/graphs/`](benchmark-campaign/report/graphs/)**.

---

## Hardware baseline

| component | spec |
|---|---|
| CPU | AMD Ryzen 9 9950X (16C/32T, boost 5.76 GHz) |
| Motherboard | ASUS ProArt X870E-Creator WiFi (AM5 / X870E) |
| RAM | 125 GiB DDR5 (+ 125 GiB zram) |
| GPU ×2 | AMD Radeon AI PRO R9700 — Navi 48, **gfx1201** (RDNA 3.5), 32 GiB VRAM, 0x7551, 300 W |
| GPU interconnect | **No xGMI** — both cards direct off the CPU; TP2 all-reduce over host SHM / PCIe |
| PCIe | Both slots PCIe 5.0 x16 @ 32 GT/s |
| OS | CachyOS (Arch), kernel 7.2.0-1-cachyos |
| Runtime | Podman 6.1.0 (rootless) + podman-compose |
| ROCm | /opt/rocm 7.2.4 |

---

## What this repo tests

| axis | options |
|---|---|
| **Backend / attention** | vLLM AITER (`ROCM_AITER_UNIFIED_ATTN`) vs vLLM RDNA4 port (`TRITON_ATTN`) vs llama.cpp |
| **Quantization** | `Qwen/Qwen3.8-27B-FP8` (stock) vs `Qwen3.8-27B-MXFP4-Quark-RDNA4` (AMD Quark) |
| **Window** | 64K / 128K / 256K context |
| **Workload** | single-user prefill+decode, 1→32-user concurrency, and a harness-agent coding task |

**The model** `Qwen3.8-27B` is a hybrid: 16 full-attention + 48 GatedDeltaNet (linear-attention)
layers — so prefill/decode cost grows with context on the attention layers, which is why
long-window decode is the interesting axis.

---

## Repo layout

```
rdna4-research/
├── benchmark-campaign/          # the apples-to-apples 4-way benchmark + full report
│   ├── report/REPORT.md         # the write-up (tables, findings, attribution)
│   ├── report/graphs/           # 9 comparison charts (g1…g9)
│   ├── data/<setup>/            # raw JSON per setup (s1/s2/s3 + vram) + agent_games/
│   ├── generate_configs.py      # builds the 6 apples-to-apples compose files
│   ├── s1_windows.py            # single-user prefill+decode @64K/128K/256K
│   ├── s2_concurrency.py        # 1→32 users, temp 0.9
│   ├── s3_agent.py              # harness-agent task (build a 3D zombie game)
│   ├── run_campaign.py          # one-command driver (benchlab start/stop + run)
│   ├── make_graphs.py / make_report.py   # regenerate report + charts from data/
│   └── README.md                # campaign-specific instructions
├── benchlab/                    # web UI to pick backend/model/params, run & visualize
├── bench_longctx.py             # standalone 128K/256K long-context bench
├── bench_concurrent.py          # concurrency stress test (temp 0.7/0.9)
├── bench_6k.py                  # 6K-prefix concurrency
├── R9700-vLLM-AB-REPORT-EN.md   # the earlier AITER-vs-TRITON A/B report (EN)
├── R9700-vLLM-AB-REPORT.md      # same report (CN)
└── RDNA4-PORT.md                # the RDNA4 port's own doc (upstream notes)
```

---

## Reproduce the benchmark

Requirements: ROCm 7.2.4, rootless podman, HF cache with the two checkpoints (FP8 + the MXFP4
export at `Capicua25x/Qwen3.8-27B-MXFP4-Quark-RDNA4`), and low-level GPU access.

> **Local config only:** `benchlab/config.json` and the generated `benchlab/benchconfigs/setup-*.yaml`
> embed machine-specific paths (your home dir, HF cache, model mount) and are **gitignored**. Copy
> [`benchlab/config.example.json`](benchlab/config.example.json) → `benchlab/config.json` and edit the
> paths (they accept `~` / `${HOME}`); `generate_configs.py` resolves paths from `$HOME` too.

```bash
# 1) generate the apples-to-apples compose files (TP2, identical flags, only backend differs)
cd rdna4-research/benchmark-campaign
./../benchvenv/bin/python generate_configs.py

# 2) start one setup (e.g. TRITON · FP8) and run the full suite
cd ../benchlab/benchconfigs && ./start-setup.sh triton-fp8 up
cd ../../benchmark-campaign
./../benchvenv/bin/python run_setup.py --setup triton-fp8 --port 8012 --model bench-triton-fp8

# 3) regenerate the report + charts from the collected data
MPLCONFIGDIR=/tmp/mpl ./../benchvenv/bin/python make_graphs.py
./../benchvenv/bin/python make_report.py
```

Or drive everything through the `benchlab` web UI (`python benchlab/app.py` → `http://127.0.0.1:8642`),
which starts/stops stacks and visualises results side by side.

---

## Apples-to-apples rule

Every setup runs the **same vLLM flags** and the **same test inputs**; only true hard
dependencies change: the attention backend, the container image, the model path, and NCCL env
(AITER needs `NCCL_P2P_DISABLE`, TRITON uses `NCCL_PROTO=Simple` — both are platform correctness
dependencies on this no-fabric PCIe box). Scenarios use thinking-off/temp-0 / thinking-on/temp-0.9
regimes as documented, and token counts always come from the server's `usage` (never SSE chunks).

---

## Attribution & licensing

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm), Apache-2.0.
- **RDNA4 port** — [Capicua25x/vllm-rocm-rdna4](https://github.com/Capicua25x/vllm-rocm-rdna4),
  Apache-2.0 (gfx1201 enablement & MXFP4 kernels; credit chain in its `RDNA4-PORT.md`:
  tcclaviger/Rob Smith, Capicua25x, andysalerno/prcoe1).
- **Models** — `Qwen/Qwen3.8-27B-FP8` and `Capicua25x/Qwen3.8-27B-MXFP4-Quark-RDNA4`
  (Apache-2.0); MXFP4 quantization via AMD Quark.
- **AI-use disclosure** — the machine, configs, and measurements are ours; AI assistance was
  used to write the runners/report. No benchmark number was generated by a model — all values
  come from recorded runs in `data/`.

---

## Roadmap / next experiments

- **1× vs 2× GPU** comparison (`--tensor-parallel-size 1`), with the caveat that a single 32 GB
  card can't hold the 262K window's KV.
- **llama.cpp** row for the FP8/MXFP4 checkpoints (current GGUFs are Q4/Q8_K_XL of the "UD" build).
- **Piecewise-cudagraph** config (the port notes `FULL_AND_PIECEWISE` hurts MTP speed at long
  context) — the current runs use default cudagraph, so long-window decode is conservative.
- More models (MoE 35B, Gemma-4, Mistral already structurally verified by the port).

---
*Generated for GitHub; methodology and exact numbers are in `benchmark-campaign/report/REPORT.md`.*
