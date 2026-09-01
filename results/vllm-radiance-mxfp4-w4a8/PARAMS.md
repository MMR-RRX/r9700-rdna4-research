# vLLM parameters used — Route B: MXFP4-W4A8 + DFlash2-FP8

Served on 2× AMD R9700, TP=2, via the radiance ROCm-10 build. This is the **B** route compared
against Route A (FP8 + DFlash2); see `report/REPORT.md`.

## Image & env

- Image: `localhost/vllm-rocm10:latest` (ROCm 10.0 build, vLLM **0.28.0**, torch **2.12.0**)
- Env (radiance MXFP4-W4A8 path):
  `RADIANCE_MXFP4=1`, `RADIANCE_MXFP4_W4A8=1`, `RADIANCE_MXFP4_DECODE_MAX_M=128`,
  `NCCL_PROTO=Simple`, `HIP_VISIBLE_DEVICES=0,1`, `HF_HUB_OFFLINE=1`.

## vLLM flags

| flag | value |
|---|---|
| model | MXFP4-W4A8 `Qwen3.8-27B` export (W4A8; served as `Qwen3.8-MXFP4`) |
| `--tensor-parallel-size` | `2` |
| `--gpu-memory-utilization` | `0.97` |
| `--max-model-len` | **`262144`** (256K) |
| `--max-num-seqs` | `8` |
| `--max-num-batched-tokens` | `8192` (chunked prefill on) |
| `--kv-cache-dtype` | `fp8` |
| `--mamba-ssm-cache-dtype` | `bfloat16` |
| `--attention-backend` | radiance R4D path (drafter uses `TRITON_ATTN`) |
| `--speculative-config` | `{"method":"dflash","model":"/models/z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":7,"attention_backend":"TRITON_ATTN"}` |
| `--enable-prefix-caching` / `--mamba-cache-mode align` / `--enable-chunked-prefill` | on |
| `--tool-call-parser qwen3_xml` / `--reasoning-parser qwen3` / `--enable-auto-tool-choice` | on |
| `--default-chat-template-kwargs` | `{"enable_thinking": true, "reasoning_effort": "xhigh"}` (as benchmarked) |
| `--override-generation-config` | `{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}` |

## What the MXFP4-W4A8 quant buys

- **Weights ~half the size of FP8** → the freed VRAM goes to the KV pool: **895,544 KV tokens** vs
  A (FP8) which is limited by weights at 163,840 max-model-len.
- **Full 256K window** (`max-model-len=262144`), verified by a cold 254,196-token prefill
  (HTTP 200, `prefill_tps=2700.3`).
