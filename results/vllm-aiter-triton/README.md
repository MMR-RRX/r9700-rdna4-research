# vLLM AITER vs TRITON — FP8 vs MXFP4 (dual R9700)

Benchmark of `Qwen3.8-27B` served on 2× AMD R9700 (TP=2) across four setups:
AITER vs the RDNA4-port `TRITON_ATTN` attention backends × FP8 vs MXFP4 quantization.

| artifact | location |
|---|---|
| **Report** | [`report/REPORT.md`](report/REPORT.md) |
| **Charts** | [`report/graphs/`](report/graphs/) (9; 128K & 256K leaderboards, concurrency, agent) |
| **Raw data** | [`data/`](data/) (per-setup JSON + agent-written games) |
| **vLLM parameters** | [`PARAMS.md`](PARAMS.md) |

**Headline:** TRITON + FP8 is best all-round (+25–40% long-context decode vs AITER; FP8 +11–15% vs
MXFP4); AITER is capped at 128K; concurrency peak 673 t/s @32.
