# vLLM parameters used — vLLM-radiance (R4D + DFlash2)

Single-config characterisation on 2× AMD R9700, TP=2, served via the radiance vLLM 0.28 fork.

## Image & environment

- Image: `magiccodingman/vllm-radiance:1.0.11` (vLLM 0.28, ROCm 7.14)
- Env: `HIP_VISIBLE_DEVICES=0,1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
  `RADIANCE_AR_MAX_KB=98304` (keeps the 84 MB prefill all-reduce on the P2P one-shot kernel),
  `NCCL_PROTO=Simple`, `TORCHINDUCTOR_COMPILE_THREADS=4`, VLLM/inductor/triton/aiter cache dirs.

## vLLM flags

| flag | value |
|---|---|
| `model` | `/models/Qwen/Qwen3.8-27B-FP8` |
| `--served-model-name` | `qwen3.8-27b` |
| `--tensor-parallel-size` | `2` |
| `--gpu-memory-utilization` | `0.97` |
| `--max-model-len` | `163840` |
| `--max-num-seqs` | `8` |
| `--max-num-batched-tokens` | `8192` (chunked prefill on) |
| `--attention-backend` | `R4D` |
| `--speculative-config` | `{"method":"dflash","model":"/models/z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":7,"attention_backend":"TRITON_ATTN"}` |
| `--enable-prefix-caching` | on |
| `--mamba-cache-mode` | `align` |
| `--topic-parser`/`--reasoning-parser` | `qwen3_coder` / `qwen3` |
| `--enable-auto-tool-choice` | on |
| `--enable-chunked-prefill` | on |
| `--enable-per-request-metrics`, `--enable-prompt-tokens-details` | on |
| `--default-chat-template-kwargs` | `{"enable_thinking": true, "reasoning_effort": "xhigh"}` |
| `--override-generation-config` | `{"temperature": 1.0, "top_p": 0.97, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}` |

## Model & drafter

- Target: `Qwen/Qwen3.8-27B-FP8` (FP8 weights, bf16 KV)
- Drafter: `z-lab/Qwen3.8-27B-DFlash2` (block-diffusion, k=7, `attention_backend=TRITON_ATTN`)

## Benchmark scenarios

1. **Content-type sweep** — single-stream decode (256 gen) across 8 categories (json, count, code,
   file_edit, math, prose, reasoning, summarization) × 2 modes: greedy+thinking-off, and
   sampled (t1.0, p0.97)+thinking-xhigh.
2. **Window sweep** — cold prefill (PP/TTFT) + resident-context decode (TG) at 8K/32K/64K/128K/160K
   (thinking off, temp 0). 160K is the max (server `max-model-len=163840`), substituting for 256K.
