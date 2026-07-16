# ML-SUPERB 1.0 & 2.0 — guide (ESPnet + frontend=s3prl, WavLM Base+)

# ML-SUPERB (1.0 and 2.0) — WavLM Base+

ML-SUPERB is the **multilingual** extension of SUPERB: instead of English-only tasks, a
**frozen** self-supervised (SSL) model is probed on speech recognition and language ID across
~143 languages, using a tiny per-language budget (10 minutes or 1 hour of training audio). As in
SUPERB, only a small head is trained on top of the frozen features, so the score reflects the
*representation quality* of the SSL model — here, **WavLM Base+**.

Both benchmarks are implemented as **ESPnet recipes** (not s3prl `run_downstream.py`). The SSL
model is loaded as an ESPnet `frontend: s3prl`, its layers are weighted-summed (the only learnable
part of the featuriser), fed through a small linear pre-encoder and a 2-layer Transformer, and
trained with **CTC** (greedy decode, **no language model**).

| | ML-SUPERB 1.0 | ML-SUPERB 2.0 |
|---|---|---|
| Recipe | `egs2/ml_superb/asr1` | `egs2/ml_superb2/asr1` |
| Paper | Interspeech 2023 (arXiv:2305.10615) + ASRU 2023 adapters (arXiv:2310.05513) | Interspeech 2024 (arXiv:2406.08641) |
| What it is here | The full multi-track benchmark (mono ASR, multi ASR, LID, joint) | The 2024 **Challenge baseline** only (one config, one split) |
| Baseline SSL in-repo | HuBERT-Large / fbank examples | Frozen **MMS-1B** |
| Data | **Manual** download (30.3 GB zip), set `MLSUPERB=` | **Auto** download from Hugging Face (~15.5 GB) |
| Metric(s) | CER (PER for cmn/jpn), LID accuracy % | Challenge scorer: Standard CER, LID, Worst-15 CER, CER StdDev, Dialect CER, Dialect LID |
| Scoring tool | `sclite` (SCTK) | custom `jiwer`-based `local/score.py` |
| Wrapper | `scripts/ml_superb/run_wavlm_mlsuperb1.sh` | `scripts/ml_superb/run_wavlm_mlsuperb2.sh` |

> **In-fork change (spec-sanctioned).** WavLM configs do not exist upstream. They were added on
> the ESPnet fork branch **`ml-superb-wavlm`** (`FabianRitter/espnet`, commit **`fd896ff`**). The
> meta-repo submodule `external/espnet` must be pinned to that commit before the wrappers can find
> the configs. Each config is a **verbatim copy** of an existing `train_asr_s3prl_*` /
> `train_mms_baseline` template with only the SSL frontend swapped (2–3 lines), which is exactly
> what the recipe README allows ("only the frontend/learning rate can be changed for the
> benchmark").

## 1. Track taxonomy — what each measures and which command runs it

**ML-SUPERB 1.0** (`egs2/ml_superb/asr1`) implements four tracks; our wrapper exposes the two ASR
tracks (the ones a WavLM row is normally reported on), and the recipe's own scripts cover the rest:

| Track | What it measures | Metric | Wrapper `--track` | Config |
|---|---|---|---|---|
| **Monolingual ASR** (10min / 1h) | Per-language ASR when the head is trained on one language's 10min/1h | **CER** (PER for cmn/jpn) | `mono10min` / `mono1h` (+`--lang`) | `train_asr_s3prl_wavlm_single.yaml` |
| **Multilingual ASR** (10min / 1h) | One head jointly trained on all languages | **CER** (macro, + few-shot split) | `multi10min` / `multi1h` | `train_asr_s3prl_wavlm_{10min,1h}.yaml` |
| **LID** | Language identification only | **LID accuracy %** | *(recipe: `run_multi.sh … --only_lid true`)* | `train_asr_s3prl_wavlm_{10min,1h}.yaml` |
| **Joint ASR+LID** | Transcribe + tag language | **CER + LID acc** | *(recipe: `run_multi.sh … --lid true`)* | `train_asr_s3prl_wavlm_{10min,1h}.yaml` |

LID and joint tracks are not wrapped (they are the same command as multilingual ASR with different
flags — see `ml_superb_1_run.md`, "Other tracks"). A **few-shot / reserved-language** result is not
a separate run: it is a scoring split (20 reserved languages held to 5 utterances each) reported
automatically inside the multilingual run as `reserved` vs `trained`.

**ML-SUPERB 2.0** (`egs2/ml_superb2/asr1`) is a **single** frozen-SSL + CTC configuration on one
`train`/`dev` split — the reproducible baseline for the 2024 Challenge. It is *not* the 2.0 paper's
full model×adaptation grid (frozen-shallow / full-FT / partial-FT / adapters), which is not codified
as recipe configs and is out of scope. Its scorer reports the six challenge metrics on the public
`dev` and `dev_dialect` splits. **The Challenge test set is held out** — every number produced here
is a **dev** number and must be labelled "dev", never as a leaderboard/test result.

## 2. Data acquisition

### ML-SUPERB 1.0 — manual download, then `MLSUPERB=`
Not auto-downloaded; the recipe only prints an instruction. Steps:

1. Download the corpus archive (one file):
   - **7th version** (Interspeech2023 / ASRU2023): Hugging Face
     `https://huggingface.co/datasets/ftshijt/mlsuperb_7th` **or** Google Drive id
     `1QYjl-7vflle__3AfuosAC5VJGiBDvEqz`. File `seventh_version.zip` = **30.3 GB**
     (30,341,235,561 bytes; unzips to ~40 GB).
   - **8th version** (Interspeech2024): HF `ftshijt/mlsuperb_8th` or GDrive
     `1vQ5NksmGl-lY7I4mlU4Kde3EhrEYGii2`.
2. Unzip. The unpacked directory **directly contains the source folders**:
   `ALFFA LAD M-AILABS NST commonvoice fleurs googlei18n_asr googlei18n_tts mexico-el mls nchlt swc voxforge voxpopuli`
   (each `<src>/<lang>/{wav/*.wav, transcript_{10min|1h}_train.txt, transcript_10min_{dev,test}.txt}`).
   Pass that directory as `--data-root`.
3. The wrapper writes it into the recipe's `db.sh` (`MLSUPERB=<path>`) for you — that is the
   recipe's documented mechanism. (`db.sh` is a symlink to `egs2/TEMPLATE/asr1/db.sh`; the wrapper
   edits the resolved file in place. This is a local, uncommitted edit — expected ESPnet workflow —
   and only ML-SUPERB 1.0 reads `MLSUPERB`.)

**Licensing:** the HF tag is `license:other`; the corpus is the union of Creative-Commons / MIT /
GNU / BSD sources, described by the authors as permissively usable for industry and academia. It is
**not gated** (no request form) but the HF dataset *viewer* fails (it is an archive dump, not
parquet) — download the archive and unzip. Note this license caveat in any redistribution.

**Fixed evaluation set:** dev/test are always the fixed **10-min** dev/test utterances regardless of
the training budget (the data-prep hard-codes `transcript_10min_{dev,test}.txt`). So a 10min-trained
and a 1h-trained model are scored on the *same* test utterances — only the directory is named by
duration (`test_1h_<lang>` for a 1h run still holds the fixed test content).

### ML-SUPERB 2.0 — automatic Hugging Face download
`local/data.sh` → `local/download.py` calls
`datasets.load_dataset("espnet/ml_superb_hf")` automatically during stage 1. The repo is **public
and ungated**, parquet-native, **~15.5 GB** total (train 12.2 GB / dev 2.3 GB / dev_dialect 1.0 GB).
Two caveats to document for users:
- The recipe hard-codes `cache_dir="."`, so the ~15.5 GB **downloads into the recipe directory**
  (`external/espnet/egs2/ml_superb2/asr1`). Budget ~16 GB of free space there plus prepared `data/`.
- Only `dev` + `dev_dialect` are public; the **Challenge test set is held out**. All numbers = dev.

## 3. Install requirements (recap — see the Infra report for the authoritative env)

Beyond `pip install -e espnet`, this recipe needs (per R2's infra brief):
- **`sclite`** (SCTK) — stage-13 scoring calls it directly (built by `tools/installers/install_sctk.sh`, part of `make all`).
- **`sox`** — every `wav.scp` line is a `sox …|` pipe (`conda install -c conda-forge sox`).
- **espnet-fork `s3prl`** — the `frontend: s3prl` loader. **ESPnet imports its *own* fork**
  (`git+https://github.com/espnet/s3prl.git`), **not** the meta-repo's vanilla `external/s3prl`
  submodule. Install it into the ESPnet env (`make s3prl.done` or the pinned `pip install`). WavLM is
  present in both forks, so `wavlm_base_plus` resolves either way — but the env must have the espnet
  fork. (This is the one genuine seam to be aware of; low risk.)
- **ML-SUPERB 2.0 only:** `datasets` (HF download) and `jiwer` (the challenge scorer).
- **Not needed:** SentencePiece/BPE, full Kaldi, LM tools (char/word tokens, `use_lm=false`).
  `pyopenjtalk`/`pypinyin` are needed only for cmn/jpn monolingual (auto-installed by the recipe).

The workspace env is `ssl-bench-espnet` (override with `SSL_BENCH_ESPNET_ENV`; on the cluster set it
to the prefix path `…/envs/ssl-bench-espnet`). Sanity check:
`python -c "import espnet2, s3prl; from s3prl.nn import S3PRLUpstream; assert 'wavlm_base_plus' in S3PRLUpstream.available_names()"` and `command -v sclite sox`.

## 4. Expected runtimes (engineering estimates — NO run performed)

All figures are **estimates** (labelled per R2's brief; anchored to the 2.0 README's "~2 days on a
single H100" for the ~10× larger frozen MMS-1B). WavLM Base+ (~95M params) is materially faster.
Actual numbers depend on hardware and GPU contention.

| Run | Iterations | Est. wall-clock (1×H100) | Notes |
|---|---|---|---|
| **mono 10min, one language** | 500 × 30 | **~0.5–1.5 h** | Cheapest meaningful run; the smoke target (use `--lang xty`, the smallest) |
| mono 1h, one language | 500 × 30 | ~0.5–1.5 h | Same head size; more train audio |
| full mono sweep (13 langs × {10min,1h}) | — | ~30–50 GPU-h | Recipe's own `./run_mono.sh` |
| **multi 10min ASR** (headline number) | 10000 × 30 | **~12–24 h** | The canonical ML-SUPERB WavLM CER |
| multi 1h ASR | 20000 × 30 | ~1.5–2.5 days | |
| LID 10min / joint 10min | 10000 × 30 | ~12–24 h each | |
| **2.0 baseline (WavLM Base+)** | 10000 × 20 | **< 1 day** | vs ~2 days for MMS-1B; +15.5 GB auto-download |

## 5. Reference-number policy (read before comparing)

**There is no published WavLM number for ML-SUPERB 1.0.** The 1.0 paper evaluated only
wav2vec2 / HuBERT / XLSR variants (best overall = XLSR-128; HuBERT-Large best on 10-min mono). Unlike
SUPERB, WavLM Base+/Large are **not** on an ML-SUPERB leaderboard. So we have **no exact figure to
reproduce** — only a **plausibility band**.

- **Sanity band (1.0), from the recipe README (HuBERT-Base, same ~95M size class, No-Adapter CER,
  10min / 1h):** eng1 **33.8 / 26.7**, deu1 **35.1 / 30.2**, jpn **20.6 / 15.6**. WavLM Base+ should
  land in a **broadly similar 20–40 CER band** per language; WavLM Large would sit nearer the
  XLSR/HuBERT-Large end. Treat as a *plausibility check, not a target*.
- **Reference row (2.0), from the recipe README (frozen MMS-1B baseline, dev):** Standard CER
  **24.0**, Standard LID **74.0**, Worst-15 CER **71.0**, CER StdDev **25.5**, Dialect CER **32.7**,
  Dialect LID **54.0**. Report the WavLM 2.0 dev numbers *against this baseline row*, clearly labelled
  dev. (Note WavLM Base+ is ~10× smaller than MMS-1B, so it need not match or beat it.)
- **Open item:** an exact WavLM-Large 2.0 CER/LID cell (the 2.0 paper *does* evaluate WavLM) could
  not be extracted from open sources in this environment — pull from the 2.0 paper PDF
  (arXiv:2406.08641) if an exact target is wanted.

## 6. Verification status (of this deliverable)

- **Configs** — created on branch `ml-superb-wavlm` (`fd896ff`); each **parses** (`yaml.safe_load`)
  and **diffs against its template to exactly the 2–3 sanctioned lines** (upstream name, preencoder
  `input_size`, and for 2.0 dropping `path_or_url`). **Verified.**
- **Wrappers** — `bash -n` clean; `--dry-run` verified for all four 1.0 tracks and the 2.0 baseline
  (exact recipe commands shown in the run docs). No cluster paths in committed files. **Verified.**
- **Training** — **not executed** in this deliverable (no GPU work). A later smoke will run the
  cheapest track (mono 10min, `--lang xty`). Until then every metric here is an **estimate/band**,
  never a result. Executed numbers, when they land, carry `SMOKE` / `REDUCED` / `FULL` labels.

# Running ML-SUPERB 1.0 (WavLM Base+)

**Prerequisites** (see `ml_superb_overview.md`): the `ssl-bench-espnet` env with `espnet2`,
espnet-fork `s3prl`, `sox`, `sclite`; the 7th-version corpus unpacked somewhere; and
`external/espnet` pinned to branch **`ml-superb-wavlm`** (`fd896ff`) so the WavLM configs exist.

One wrapper drives everything: **`scripts/ml_superb/run_wavlm_mlsuperb1.sh`**. It selects the track,
writes your corpus path into the recipe's `db.sh`, activates the env, and runs ESPnet stages 1→13
(data-prep → token list → collect-stats → train → CTC-greedy decode → `sclite` scoring). The
frozen WavLM Base+ checkpoint (~0.4 GB) auto-downloads to `<recipe>/hub` on first use.

```
--track   mono10min | mono1h | multi10min | multi1h   (required)
--data-root PATH   unpacked 7th-version corpus dir (required; see overview §2)
--lang LANG        mono only; default xty; one of
                   eng1 eng2 eng3 fra1 fra2 deu1 deu2 rus swa swe jpn cmn xty
--asr-config PATH  override the training YAML (default: the track's WavLM config)
--stage / --stop-stage   ESPnet stage range (default 1 / 13)
--gpu IDS          CUDA_VISIBLE_DEVICES (default 0)
--exp-suffix S     provenance tag on the wrapper log filename
--dry-run          print the exact recipe command and exit
```

Always try `--dry-run` first — it prints the precise `run.sh` / `run_multi.sh` command and the
`MLSUPERB=` edit, without touching anything.

## Cheapest real run — monolingual 10-min, one language (smoke target)

```bash
bash scripts/ml_superb/run_wavlm_mlsuperb1.sh \
     --track mono10min --lang xty \
     --data-root /path/to/seventh_version_unpacked
```
Runs (under the hood):
```
./run.sh --stage 1 --stop_stage 13 --multilingual false --single_lang xty \
         --duration 10min --lid false --only_lid false \
         --asr_config conf/tuning/train_asr_s3prl_wavlm_single.yaml
```
> **Why `--lid false --only_lid false` is forced:** the recipe's `run.sh` defaults `only_lid=true`,
> which would make stage-13 `local/score.sh` attempt **LID** scoring and abort on a monolingual
> reference (no `[iso]` language tags → an assertion failure). Forcing both false selects the plain
> **CER** path. (This is a correction to the bare command in R2's brief; the wrapper handles it.)

**Read the result** at:
```
external/espnet/egs2/ml_superb/asr1/exp/train_asr_s3prl_wavlm_single_xty_10min/
    decode_asr_asr_model_valid.loss.ave.pth/test_10min_xty/score_cer/result.txt
```
Open `result.txt` and read the `Sum/Avg` row: the **Err** column is the **CER %** (sclite reports
substitution+deletion+insertion over characters; lower is better). `exp/<tag>/RESULTS.md` gives a
Markdown summary of the same. For `mono1h`, the tag is `…_single_xty_1h` and the dset dir is
`test_1h_xty` (same fixed test utterances — see overview §2).

> **cmn / jpn:** these two are officially scored as **PER** (phoneme error rate, via word/g2p
> tokenisation) and only by the recipe's `./run_mono.sh`. The wrapper's single-language path uses
> char tokens (CER) and prints a warning for cmn/jpn. For the official cmn/jpn number, run
> `./run_mono.sh` directly (next section).

## Headline number — multilingual 10-min ASR

```bash
bash scripts/ml_superb/run_wavlm_mlsuperb1.sh \
     --track multi10min \
     --data-root /path/to/seventh_version_unpacked
```
Runs:
```
./run_multi.sh --stage 1 --stop_stage 13 --duration 10min --lid false --only_lid false \
               --asr_config conf/tuning/train_asr_s3prl_wavlm_10min.yaml
```
**Read the result** at:
```
external/espnet/egs2/ml_superb/asr1/exp/train_asr_s3prl_wavlm_10min_multilingual_10min/
    RESULTS.md                                                    # human-readable summary
    decode_asr_asr_model_valid.loss.ave.pth/<dset>/score_cer/few_shot/trained/result.txt   # main CER
    decode_asr_asr_model_valid.loss.ave.pth/<dset>/score_cer/few_shot/reserved/result.txt  # few-shot langs
```
The **`trained`** bucket is the standard multilingual CER (languages seen in training); **`reserved`**
is the zero-/few-shot CER on the 20 held-out languages. `multi1h` is identical with
`--track multi1h` and the `…_1h` config/tag.

## Other 1.0 tracks (LID, joint ASR+LID, full mono sweep) — recipe-native

These are one flag away from the multilingual command; run them directly in the recipe
(`cd external/espnet/egs2/ml_superb/asr1` with the env active and `MLSUPERB` already set by any prior
wrapper call):

```bash
# LID (accuracy %)         — token_type auto-switches to word
./run_multi.sh --asr_config conf/tuning/train_asr_s3prl_wavlm_10min.yaml \
               --duration 10min --lid false --only_lid true
#   result: decode_*/<dset>/score_lid/few_shot/{trained,reserved}/scores.txt  ("Acc: XX.XX%")

# Joint ASR + LID          — emits "[lang] text"
./run_multi.sh --asr_config conf/tuning/train_asr_s3prl_wavlm_10min.yaml \
               --duration 10min --lid true --only_lid false
#   result: both score_cer/ (CER) and score_lid/ (LID acc) splits

# Full monolingual sweep   — all 13 languages × {10min,1h}, official aggregation
./run_mono.sh --asr_config conf/tuning/train_asr_s3prl_wavlm_single.yaml
#   result: exp/mono_train_asr_s3prl_wavlm_single.log
#           ("Average Error Rate (10min):…" and "(1h):…", macro-averaged)
```

## Tips
- **Resume / re-score without retraining:** `--stage 12` (decode only) or `--stage 13` (score only);
  e.g. `--stage 13 --stop-stage 13` re-runs just `sclite`.
- **Data-prep only:** `--stop-stage 1` (writes `data/…`; also performs the `MLSUPERB=` edit).
- **Logs:** the wrapper tees the whole run to `logs/ml_superb/mlsuperb1_<track>[_<lang>][_<suffix>].log`
  (relative to the repo). ESPnet's own per-stage logs live under `exp/<tag>/`.
- **Checkpoint:** scored model is the averaged `valid.loss.ave.pth` (best on the fixed dev set).

# Running ML-SUPERB 2.0 (WavLM Base+ — 2024 Challenge baseline)

This reproduces the **Interspeech-2024 ML-SUPERB 2.0 Challenge baseline** with WavLM Base+ swapped in
for the stock MMS-1B. It is a **single** frozen-SSL + CTC configuration on one `train`/`dev` split —
not the 2.0 paper's full grid. **Scoring is dev-only** (the Challenge test set is held out), so every
number is a **dev** number; label it as such.

**Prerequisites** (see `ml_superb_overview.md`): the `ssl-bench-espnet` env with `espnet2`,
espnet-fork `s3prl`, `sox`, `sclite`, **`datasets`** (HF) and **`jiwer`**; and `external/espnet`
pinned to branch **`ml-superb-wavlm`** (`fd896ff`) so `conf/tuning/train_wavlm_baseline.yaml` exists.

Wrapper: **`scripts/ml_superb/run_wavlm_mlsuperb2.sh`**.

```
--asr-config PATH   training YAML (default conf/tuning/train_wavlm_baseline.yaml)
--stage / --stop-stage   ESPnet stage range (default 1 / 13; stage 1 does the HF download)
--gpu IDS           CUDA_VISIBLE_DEVICES (default 0)
--exp-suffix S      provenance tag on the wrapper log filename
--dry-run           print the exact recipe command and exit
```

## Data — automatic, ~15.5 GB, no manual step

Unlike 1.0, **there is no `--data-root`**. Stage 1 auto-downloads
`datasets.load_dataset("espnet/ml_superb_hf")` — a **public, ungated** HF dataset, **~15.5 GB**
(train 12.2 GB / dev 2.3 GB / dev_dialect 1.0 GB) — and prepares `data/{train,dev,dev_dialect}`.
Two things to know:
- The recipe hard-codes `cache_dir="."`, so the raw data **caches into the recipe directory**
  (`external/espnet/egs2/ml_superb2/asr1`). Ensure ~16 GB free there (plus prepared `data/`).
- `dev_dialect` is the 2.0 dialect-robustness eval. The **Challenge test set is not shipped** — all
  results are on `dev` / `dev_dialect`.

## Run it (one command)

```bash
bash scripts/ml_superb/run_wavlm_mlsuperb2.sh
```
Runs (under the hood):
```
./run.sh --asr_config conf/tuning/train_wavlm_baseline.yaml --stage 1 --stop_stage 13
```
> `run.sh` forwards extra arguments to `asr.sh`, and ESPnet's option parser is **last-wins**, so the
> appended `--asr_config` overrides the recipe's built-in default (`conf/train_asr.yaml`, the MMS-1B
> baseline). This was verified explicitly.

Expected wall-clock: **< 1 day on a 1×H100** (engineering estimate; the stock MMS-1B baseline is
"~2 days" per the recipe README, and WavLM Base+ is ~10× smaller). If a 40 GB GPU OOMs, halve
`batch_size` and double `accum_grad` in the config (the README's own guidance).

## Read the result

Stage 13 auto-runs the custom scorer (`local/score.py`) and writes:
```
external/espnet/egs2/ml_superb2/asr1/exp/<asr_tag>/challenge_results.md
```
with the six challenge columns:

| Standard CER | Standard LID | Worst-15 CER | CER StdDev | Dialect CER | Dialect LID |
|---|---|---|---|---|---|

(macro-averaged over languages; CER via `jiwer` after punctuation-strip + uppercasing, plus
space-removal for cmn/jpn/tha/yue). Re-score any existing model folder without retraining:
```bash
cd external/espnet/egs2/ml_superb2/asr1
python local/score.py --exp_dir exp/<asr_tag>
```
The hypotheses it reads live under
`exp/<asr_tag>/decode_asr_asr_model_valid.loss.ave/{org/dev,dev_dialect}/text`.

## Compare against the baseline (label everything "dev")

| Row | Standard CER | Standard LID | Worst-15 CER | CER StdDev | Dialect CER | Dialect LID |
|---|---|---|---|---|---|---|
| **MMS-1B baseline** (README, dev) | 24.0 | 74.0 | 71.0 | 25.5 | 32.7 | 54.0 |
| WavLM Base+ (our run) | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |

WavLM Base+ is ~10× smaller than MMS-1B, so it need not match the baseline — the point is a
**reproducible, honestly-labelled 2.0 dev artifact**, not a leaderboard entry. If an exact WavLM 2.0
target is required, extract it from the 2.0 paper PDF (arXiv:2406.08641); it was not machine-readable
in this environment.

## Tips
- **Resume:** `--stage 11` (train onward) / `--stage 12` (decode) / `--stage 13` (score only).
- **Logs:** teed to `logs/ml_superb/mlsuperb2_baseline[_<suffix>].log`.
- **Checkpoint:** averaged `valid.loss.ave.pth`.
