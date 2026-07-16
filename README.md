# SSL-models-benchmarking

Unified, reproducible benchmarking of self-supervised (SSL) speech models across three
community benchmarks that were never designed to share a codebase:

- **SUPERB** (Interspeech 2021) — 10 speech-understanding tasks, implemented in **s3prl**.
- **SUPERB-SG** (ACL 2022) — 4 semantic & generative tasks, also in **s3prl**.
- **ML-SUPERB 1.0 & 2.0** — multilingual ASR + language identification, implemented in
  **ESPnet** and driven through ESPnet's built-in `frontend=s3prl` integration.

**WavLM Base+** is wired up as the worked example on every task, so a non-expert can
follow one coherent workflow to produce a benchmark number for any s3prl upstream model
on any of the three benchmarks. This is a reproducibility and documentation effort, not
a research contribution.

> **Scope & honesty.** Every wrapper script's *default* invocation is the full,
> paper-faithful benchmark run — that is what you execute to get real numbers. What has
> been *verified in this repo* is documented per task in
> [`results/RESULTS.md`](results/RESULTS.md): tasks labeled **SMOKE** were executed
> end-to-end on real data with reduced steps (pipeline proof — those metric values are
> meaningless as benchmarks); tasks labeled **DOCUMENTED** are dry-run verified. No
> benchmark numbers in this repository are fabricated; reference numbers are always
> attributed to their source.

## Design: one meta-repo + two pinned forks

s3prl (PyTorch) and ESPnet interoperate only through ESPnet's `frontend=s3prl` seam
(any ESPnet ASR recipe can consume an s3prl upstream as its feature extractor). Rather
than merge the frameworks, this repository holds documentation and wrapper scripts and
consumes both upstreams as pinned git submodules:

| Submodule | Fork | Pinned at | Role |
|---|---|---|---|
| `external/s3prl` | [FabianRitter/s3prl](https://github.com/FabianRitter/s3prl) (vanilla `s3prl/s3prl` content) | `ec8064b` | SUPERB + SUPERB-SG |
| `external/espnet` | [FabianRitter/espnet](https://github.com/FabianRitter/espnet), branch `ml-superb-wavlm` | `fd896ff` | ML-SUPERB 1.0 + 2.0 |

The espnet pin is upstream `6ed85c0` **plus one commit** adding the WavLM Base+ tuning
configs to both ML-SUPERB recipes (`egs2/ml_superb/asr1/conf/tuning/train_asr_s3prl_wavlm_{single,10min,1h}.yaml`,
`egs2/ml_superb2/asr1/conf/tuning/train_wavlm_baseline.yaml`) — the normal way of
contributing a new frontend config to an ESPnet recipe. ML-SUPERB's logic is never
vendored into s3prl.

## Install

```bash
git clone --recursive https://github.com/FabianRitter/SSL-models-benchmarking.git
cd SSL-models-benchmarking
```

**Environment 1 — s3prl side (SUPERB / SUPERB-SG):**

```bash
conda create -n ssl-bench-s3prl python=3.10 -y && conda activate ssl-bench-s3prl
pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e "./external/s3prl[all]"
pip install "transformers==4.46.3"   # REQUIRED: s3prl leaves transformers unpinned; 5.x breaks under torch 2.4
```

Known issue on Python ≥3.10: s3prl's unpinned `catalyst` dependency (imported only by
the KS task) crashes on import (`collections.MutableMapping` was removed). The
[KS section of docs/superb.md](docs/superb.md) documents a 10-line `sitecustomize.py`
workaround.

**Environment 2 — ESPnet side (ML-SUPERB):**

```bash
cd external/espnet/tools
./setup_anaconda.sh "$(conda info --base)" ssl-bench-espnet 3.10
make TH_VERSION=2.4.1 CUDA_VERSION=12.1   # espnet + sctk (sclite) etc.
make s3prl.done                            # ESPnet's own s3prl fork — required by frontend=s3prl
cd ../../..
```

`sox` must be on PATH (e.g. `conda install -c conda-forge sox`). ML-SUPERB's recipes use
character tokens with CTC-greedy decoding — no Kaldi binaries, no SentencePiece training,
no LM tools are needed.

## Quickstart (one task end-to-end)

Keyword Spotting is the cheapest full benchmark run (~4–12 h on one modern GPU):

```bash
# 1. data (~2.1 GB, public)
mkdir -p data/ks/speech_commands_v0.01 data/ks/speech_commands_test_set_v0.01
wget -qO- http://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz | tar xz -C data/ks/speech_commands_v0.01
wget -qO- http://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz | tar xz -C data/ks/speech_commands_test_set_v0.01
# 2. run (train + evaluate; prints "RESULT superb ks acc=...")
bash scripts/superb/run_wavlm_ks.sh --data-root data/ks --gpu 0
```

Every wrapper shares the same flags: `--data-root`, `--upstream` (default
`wavlm_base_plus`), `--gpu`, `--exp-name`, `--stage`, `--extra-override`, `--dry-run`.
To benchmark a different s3prl upstream, change `--upstream` — nothing else.

## Task index

| Benchmark | Task | Metric | Reference (WavLM Base+) | Script | Docs |
|---|---|---|---|---|---|
| SUPERB | PR — Phoneme Recognition | PER ↓ | 3.92 | `scripts/superb/run_wavlm_pr.sh` | [docs/superb.md](docs/superb.md) |
| SUPERB | ASR | WER ↓ (no LM) | 5.59 | `scripts/superb/run_wavlm_asr.sh` | [docs/superb.md](docs/superb.md) |
| SUPERB | KS — Keyword Spotting | Acc ↑ | 97.37 | `scripts/superb/run_wavlm_ks.sh` | [docs/superb.md](docs/superb.md) |
| SUPERB | QbE — Query by Example | MTWV ↑ | 0.0988 | `scripts/superb/run_wavlm_qbe.sh` | [docs/superb.md](docs/superb.md) |
| SUPERB | SID — Speaker ID | Acc ↑ | 89.42 | `scripts/superb/run_wavlm_sid.sh` | [docs/superb.md](docs/superb.md) |
| SUPERB | ASV — Speaker Verification | EER ↓ | 4.07 | `scripts/superb/run_wavlm_asv.sh` | [docs/superb.md](docs/superb.md) |
| SUPERB | SD — Diarization | DER ↓ | 3.50 | `scripts/superb/run_wavlm_sd.sh` | [docs/superb.md](docs/superb.md) |
| SUPERB | IC — Intent Classification | Acc ↑ | 99.00 | `scripts/superb/run_wavlm_ic.sh` | [docs/superb.md](docs/superb.md) |
| SUPERB | SF — Slot Filling | F1 ↑ / CER ↓ | 90.58 / 21.20 | `scripts/superb/run_wavlm_sf.sh` | [docs/superb.md](docs/superb.md) |
| SUPERB | ER — Emotion Recognition | Acc ↑ (5-fold mean) | 68.65 | `scripts/superb/run_wavlm_er.sh` | [docs/superb.md](docs/superb.md) |
| SUPERB-SG | SE — Speech Enhancement | PESQ/STOI ↑ | (HuBERT-L ballpark 2.64/94.2)* | `scripts/superb_sg/run_wavlm_se.sh` | [docs/superb_sg.md](docs/superb_sg.md) |
| SUPERB-SG | SS — Source Separation | SI-SDRi ↑ | (ballpark 10.45)* | `scripts/superb_sg/run_wavlm_ss.sh` | [docs/superb_sg.md](docs/superb_sg.md) |
| SUPERB-SG | VC — Voice Conversion (a2o) | MCD ↓ | (ballpark 7.22)* | `scripts/superb_sg/run_wavlm_vc_a2o.sh` | [docs/superb_sg.md](docs/superb_sg.md) |
| SUPERB-SG | ST — Speech Translation | BLEU ↑ | (ballpark 20.01)* | `scripts/superb_sg/run_wavlm_st.sh` | [docs/superb_sg.md](docs/superb_sg.md) |
| ML-SUPERB 1.0 | mono/multi ASR + LID, 10min & 1h | CER ↓ / LID Acc ↑ | (no published WavLM row)* | `scripts/ml_superb/run_wavlm_mlsuperb1.sh` | [docs/ml_superb.md](docs/ml_superb.md) |
| ML-SUPERB 2.0 | frozen-SSL CTC baseline (dev) | CER/WER ↓, LID ↑ | (MMS-1B baseline, dev)* | `scripts/ml_superb/run_wavlm_mlsuperb2.sh` | [docs/ml_superb.md](docs/ml_superb.md) |

\* SUPERB-SG's paper did not evaluate WavLM and the public leaderboard was unreachable
when this repo was built; ML-SUPERB's papers contain no WavLM row at all. The docs give
the closest honest anchors instead — never treat them as WavLM targets.

Per-task verification status: [`results/RESULTS.md`](results/RESULTS.md).

## Data you must obtain yourself (gated / large)

Documented per task in the docs pages, with request URLs: **IEMOCAP** (USC form, ER),
**Fluent Speech Commands** (fluent.ai signup or HF backup, IC), **VoxCeleb1** (~34 GB,
SID/ASV), **Common Voice Corpus 4 en** (Mozilla account — the ST blocker, together with
a working `fairseq` install), **ML-SUPERB 1.0 release** (~30 GB archive), **ML-SUPERB
2.0** (auto-downloads ~15.5 GB from Hugging Face during data prep). SD and SS *generate*
Libri2Mix locally from LibriSpeech + WHAM! noise.

## Repository layout

```
SSL-models-benchmarking/
├── README.md              # this file
├── LICENSE                # Apache-2.0 (matches both upstreams)
├── docs/                  # task-by-task guides (superb, superb_sg, ml_superb)
├── scripts/               # run_wavlm_<task>.sh wrappers (generic bash + conda; no scheduler assumptions)
├── results/RESULTS.md     # per-task verification status + reference numbers
└── external/              # pinned submodules: s3prl, espnet (see table above)
```

## Attribution & citations

This work builds directly on the benchmarks and toolkits below; please cite the
originals if you use it:

- **SUPERB** — Yang et al., *SUPERB: Speech processing Universal PERformance Benchmark*, Interspeech 2021 (arXiv:2105.01051).
- **SUPERB-SG** — Tsai et al., *SUPERB-SG: Enhanced Speech processing Universal PERformance Benchmark for Semantic and Generative Capabilities*, ACL 2022 (arXiv:2203.06849).
- **ML-SUPERB** — Shi et al., Interspeech 2023 (arXiv:2305.10615); **ML-SUPERB 2.0** — Shi et al., Interspeech 2024 (arXiv:2406.08641).
- **s3prl** — the S3PRL Speech Toolkit — https://github.com/s3prl/s3prl.
- **ESPnet** — Watanabe et al., *ESPnet: End-to-End Speech Processing Toolkit* — https://github.com/espnet/espnet.
- **WavLM** — Chen et al., *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing*, IEEE JSTSP 2022 (arXiv:2110.13900).

Licensed under the Apache License 2.0 (see [`LICENSE`](LICENSE)), matching both s3prl and ESPnet.
