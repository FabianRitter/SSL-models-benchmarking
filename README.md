# SSL-models-benchmarking

> **Status: work in progress.** This repository is being built out task-by-task.
> A script and its documentation are marked verified only once they have actually
> been executed and the resulting metric recorded. Any task without a recorded run
> is explicitly labelled as such. No benchmark numbers in this repository are
> fabricated, and reference numbers are always attributed to their source.

Unified, reproducible benchmarking of self-supervised (SSL) speech models across
three community benchmarks that were never designed to share a codebase:

- **SUPERB** (Interspeech 2021) — speech-understanding tasks, implemented in **s3prl**.
- **SUPERB-SG** (ACL 2022) — semantic & generative tasks plus robustness variants, also in **s3prl**.
- **ML-SUPERB / ML-SUPERB 2.0** — multilingual ASR + language identification (LID),
  implemented in **ESPnet** (`egs2/ml_superb/asr1`) and driven through ESPnet's
  built-in `frontend=s3prl` integration.

**WavLM Base+** is run end-to-end on every runnable task as the worked example, so a
non-expert can follow one coherent workflow to reproduce a number for any s3prl
upstream model on any of the three benchmarks. This is a reproducibility and
documentation effort, not a research contribution.

## Design: one meta-repo + two pinned forks

s3prl (PyTorch) and ESPnet are separate frameworks that interoperate only through
ESPnet's `frontend=s3prl` seam (any ESPnet ASR recipe can consume an s3prl upstream
as its feature extractor). Rather than merge them, this repository is a **meta-repo**:
it holds the documentation, wrapper scripts, and setup notes, and consumes two
upstream repositories as pinned git submodules under `external/`:

| Submodule | Upstream (fork target) | Pinned commit | Role |
|---|---|---|---|
| `external/s3prl` | https://github.com/s3prl/s3prl | `ec8064b` | SUPERB + SUPERB-SG |
| `external/espnet` | https://github.com/espnet/espnet | `6ed85c0` | ML-SUPERB |

The only expected in-fork change is a WavLM tuning config (plus a reproduction
script) added under ESPnet's ML-SUPERB recipe — a normal recipe-repo contribution.
ML-SUPERB's logic is never vendored into s3prl; the `frontend=s3prl` seam is respected.

> The submodules are not wired up yet. The public forks are created, and their URLs
> pinned, as part of the first public release. See [`external/README.md`](external/README.md).

## Repository layout

```
SSL-models-benchmarking/
├── README.md              # this file
├── LICENSE                # Apache-2.0 (matches both upstreams)
├── docs/
│   ├── superb.md          # SUPERB task-by-task guide
│   ├── superb_sg.md       # SUPERB-SG task-by-task guide
│   └── ml_superb.md       # ML-SUPERB guide (ESPnet + frontend=s3prl)
├── scripts/
│   ├── superb/            # run_wavlm_<task>.sh wrappers (generic local/conda)
│   ├── superb_sg/
│   └── ml_superb/
├── results/
│   └── RESULTS.md         # cross-benchmark results table (full/reduced labelled)
└── external/
    └── README.md          # the two pinned submodules + fork plan
```

## Quickstart

_Full install and per-task run instructions are added here as each task is verified._
Two conda environments are used — one for the s3prl side (SUPERB / SUPERB-SG) and one
for the ESPnet side (ML-SUPERB); each installs a CUDA-enabled PyTorch and
editable-installs its upstream. Wrapper scripts are plain bash + `conda activate`,
with GPU selection via `CUDA_VISIBLE_DEVICES` or a `--gpu` flag — no cluster
scheduler assumptions. See the per-benchmark pages under `docs/` for detail.

## Attribution & citations

This work builds directly on the benchmarks and toolkits below; please cite the
originals if you use it:

- **SUPERB** — Yang et al., *SUPERB: Speech processing Universal PERformance Benchmark*, Interspeech 2021 (arXiv:2105.01051).
- **SUPERB-SG** — Tsai et al., *SUPERB-SG: Enhanced Speech processing Universal PERformance Benchmark for Semantic and Generative Capabilities*, ACL 2022 (arXiv:2203.06849).
- **ML-SUPERB** — Shi et al., *ML-SUPERB: Multilingual Speech Universal PERformance Benchmark*, Interspeech 2023 (arXiv:2305.10615); and **ML-SUPERB 2.0** (arXiv:2406.08641).
- **s3prl** — the S3PRL Speech Toolkit — https://github.com/s3prl/s3prl.
- **ESPnet** — Watanabe et al., *ESPnet: End-to-End Speech Processing Toolkit* — https://github.com/espnet/espnet.
- **WavLM** — Chen et al., *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing*, IEEE JSTSP 2022 (arXiv:2110.13900).

Licensed under the Apache License 2.0 (see [`LICENSE`](LICENSE)), matching both s3prl
and ESPnet.
