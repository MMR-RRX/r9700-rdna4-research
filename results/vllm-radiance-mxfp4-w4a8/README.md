# vLLM-radiance (R4D) + MXFP4-W4A8 + DFlash2-FP8 — dual R9700

Benchmark of `Qwen3.8-27B` served on 2× AMD R9700 (TP=2) via the radiance vLLM 0.28 fork with the
**MXFP4-W4A8** weight quant and the DFlash2 drafter — the **Route B** upgrade over the FP8 target.

| artifact | location |
|---|---|
| **Report (A vs B)** | [`report/REPORT.md`](report/REPORT.md) |
| **Raw data** | [`data/`](data/) (`content_sweep.json`, `full_context.json`) |
| **vLLM parameters** | [`PARAMS.md`](PARAMS.md) |

**Headline:** B is faster *and* bigger — **+19–43% production decode** (sampled, thinking xhigh) across
all content types, a **full 256K window** (verified 254K accepted, 2,700 t/s prefill) vs A's 160K, and
**895,544 KV tokens**. See the report for the caveats (greedy short-output cells and a prefix-cache
artifact on one prefill row).
