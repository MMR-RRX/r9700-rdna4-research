# vLLM-radiance (R4D) + DFlash2 — dual R9700

Benchmark of `Qwen3.8-27B` served on 2× AMD R9700 (TP=2) via the radiance vLLM 0.28 fork
(R4D attention) with the DFlash2 block-diffusion drafter.

| artifact | location |
|---|---|
| **Report** | [`report/REPORT.md`](report/REPORT.md) |
| **Charts** | [`report/figs/`](report/figs/) (5 scatter plots) |
| **Raw data** | [`data/`](data/) (`content_sweep.json`, `window_sweep.json`) |
| **vLLM parameters** | [`PARAMS.md`](PARAMS.md) |

**Headline:** decode is content-dependent — 185–193 t/s on structured/deterministic output, ~64 t/s on
open-ended prose — and holds ~150–165 t/s even at 128K/160K resident. Prefill falls 2.9k → 2.1k t/s as
context grows 8K → 160K (the server's max-model-len, substituting for 256K).
