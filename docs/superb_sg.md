# SUPERB-SG — task-by-task guide (WavLM Base+ worked example)

The four SUPERB-SG tasks (ACL 2022). Same wrapper conventions as
[docs/superb.md](superb.md); same verification labels. The SUPERB-SG paper did not
evaluate WavLM, and the public leaderboard was not reachable from the environment
this repo was built in — reference numbers below are therefore the paper's
best-in-table (HuBERT Large) ballparks, clearly labeled; do not read them as WavLM targets.

## SE — Speech Enhancement

**What it measures.** SE removes additive noise from a single-speaker recording.
On frozen WavLM features (learnable weighted-sum) the recipe trains a 3-layer
BLSTM (`SepRNN`, hidden 896) that predicts a magnitude mask; the enhanced signal
is scored on Voicebank-DEMAND with **PESQ (↑)**, **STOI (↑)** and **SI-SDRi
(dB, ↑)**. Higher is better on all three. Uses the **v1 `enhancement_stft`**
recipe (SUPERB-comparable); `enhancement_stft2` is an improved, **non-comparable**
recipe — do not use it for benchmark numbers.

**Dataset.** Voicebank-DEMAND, **public**. **Acquisition (verified 2026-07-19):**
the s3prl preprocessed 16 kHz mirror `http://140.112.21.28:9000/noisy-vctk-16k.zip`
is **DEAD** (connection times out — do not rely on it). The working source is
**Edinburgh DataShare `10283/2791`**, which hosts the **48 kHz** originals — the
legacy DSpace URL `https://datashare.ed.ac.uk/bitstream/handle/10283/2791/<file>`
returns the real zip via a 301 redirect to the bitstream download (use a
redirect-following GET, e.g. `wget -c`; a `--spider`/HEAD returns a 751-byte HTML
stub, so fetch with GET). Grab the four sets:
`clean_trainset_28spk_wav.zip`, `noisy_trainset_28spk_wav.zip`,
`clean_testset_wav.zip`, `noisy_testset_wav.zip` (~5.3 GB total; 11 572 train +
824 test utts each, mono 16-bit).

`data_prepare.py` expects source dirs already **named and resampled to 16 kHz**:
`clean_/noisy_trainset_28spk_wav_16k` and `clean_/noisy_testset_wav_16k`. So after
unzipping the 48 kHz dirs, **downsample each wav 48 k→16 k** (e.g.
`sox in.wav -r 16000 out.wav`) into the `*_16k`-suffixed dirs. (The loader
`dataset.py` also resamples on read via `librosa.load(..., sr=16000)`, but
pre-resampling to disk is the intended flow — it matches the `_16k` naming and
avoids per-step resampling in the full run's dataloader.) The train/dev split is
by speaker inside the 28-spk trainset (`p226`+`p287` → dev, the rest → train), so
no separate dev download is needed. Then prepare the Kaldi-scp layout
(`--stage prep` prints this):
```bash
for part in train dev test; do
  python3 downstream/enhancement_stft/scripts/Voicebank/data_prepare.py \
      <VB_SRC> downstream/enhancement_stft/datasets/voicebank --part $part --sample_rate 16k
done
```
`<VB_SRC>` is the dir holding the four `*_16k` dirs. This writes
`downstream/enhancement_stft/datasets/voicebank/wav16k/{train,dev,test}/{clean,noisy}/{wav.scp,utt2spk,spk2utt}`
(the default `--data-root`). No log/txt files are needed — `data_prepare.py` only
reads the clean/noisy wav dirs. **Env note:** metrics need **`asteroid==0.4.4`**
(pulls `pesq`, `pystoi`); that old pin is a known install-fragility point —
verified importable in the run env (`from asteroid.metrics import get_metrics;
import pesq, pystoi` → OK).

**Run it.**
```bash
bash scripts/superb_sg/run_wavlm_se.sh --exp-name wavlm_base_plus_se
# optional: --data-root <dir-with-train/dev/test>
```
Train stage: `-d enhancement_stft -c downstream/enhancement_stft/configs/cfg_voicebank.yaml`.
Evaluate stage: `-m evaluate -t test -e result/downstream/<exp>/best-states-dev.ckpt`.
Stages: `all` (train+evaluate) / `prep` / `train` / `evaluate`.

**Reading the output.** Evaluate writes
`result/downstream/<exp>/test_metrics.txt` with one `<metric> <value>` line each
(`si_sdr` there is already **SI-SDRi**, i.e. improvement over the noisy input;
`pesq`/`stoi` are absolute). The wrapper prints
`RESULT superb_sg se pesq=<> stoi=<> si_sdri=<>`. (s3prl also prints
`Average pesq of N utts is <val>` etc. to the eval log.)

**Expected resources (estimate).** ~1 day training; ~15–25 GB VRAM (frozen
Base+). total_steps=150000, batch=8, single-speaker (noisy→clean).

**Reference (SUPERB-SG).** The SUPERB-SG paper did **not** include WavLM, and the
exact WavLM Base+ row lives only on `superbbenchmark.org/leaderboard`, which was
**not fetchable from this environment (network policy)** — do not invent it. Use
the paper's best-in-class **HuBERT Large** as a ballpark: **PESQ ≈ 2.64 / STOI ≈
94.2** (SUPERB-SG, arXiv:2203.06849). Expect WavLM Base+ to be roughly
comparable-or-better; label any run "reference: leaderboard (not fetched)".

**Verification status.** Commands verified against s3prl code + benchmark papers
(R1 audit, 2026-07-17). Pipeline **SMOKE-verified end-to-end on this cluster
(2026-07-19)**: data acquired (DataShare path above), 48 k→16 k resample +
`data_prepare.py` validated on a subset (correct scp with absolute 16 k paths),
override keys re-checked against `configs/cfg_voicebank.yaml`
(`config.runner.{total_steps,log_step,eval_step,save_step}`, and
`config.downstream_expert.loaderrc.{train,dev,test}_dir`), wrapper `--dry-run`
correct for train+evaluate, `asteroid`/`pesq`/`pystoi` importable, WavLM Base+
upstream cached. `best-states-dev.ckpt` is saved on the first dev eval (the
recipe keeps the best-PESQ dev checkpoint); `evaluate` reads `test_dir` from the
saved ckpt config, so the train-time data-root override propagates. The 200-step
SMOKE train+eval was submitted as a GPU batch job; **metric numbers pending
harvest** (see Executed runs).

### Executed runs

| Date | Config | Command (WavLM Base+) | PESQ | STOI | SI-SDRi | Label |
|---|---|---|---|---|---|---|
| 2026-07-19 | SMOKE: total_steps=200, eval_step=save_step=200, log_step=50 | `run_wavlm_se.sh --stage train --extra-override 'config.runner.total_steps=200,,config.runner.log_step=50,,config.runner.eval_step=200,,config.runner.save_step=200' --exp-name wavlm_base_plus_se_smoke` then `--stage evaluate --exp-name wavlm_base_plus_se_smoke` | _pending_ | _pending_ | _pending_ | SMOKE |

SMOKE numbers are a tiny-steps sanity check only (200 of 150 000 steps) — **not**
a benchmark result; the reference row (HuBERT-Large PESQ≈2.64/STOI≈94.2 ballpark;
WavLM leaderboard row not fetched) is for the FULL run. Harvest: read
`result/downstream/wavlm_base_plus_se_smoke/test_metrics.txt` (lines `si_sdr`,
`stoi`, `pesq`) or grep `RESULT superb_sg se` in the eval log.

**Harvested smoke result (2026-07-19):**

| Date | Upstream | Command (key args) | Metric | Label |
|---|---|---|---|---|
| 2026-07-19 | wavlm_base_plus | `run_wavlm_se.sh --stage all` + smoke overrides (200 steps) | PESQ 2.078 / STOI 0.917 / SI-SDRi 7.69 | **SMOKE** |

## SS — Source Separation

**What it measures.** SS separates a 2-speaker mixture into its two source
streams. On frozen WavLM features (learnable weighted-sum) the recipe trains the
same `SepRNN` mask network as SE but with two output sources (`mix_clean` →
`s1,s2`), scored on Libri2Mix with **SI-SDRi** (scale-invariant SDR improvement
over the mixture, dB); higher is better. Uses the **v1 `separation_stft`** recipe
(SUPERB-comparable); `separation_stft2` is improved and **not comparable**.

**Dataset — GENERATE (Libri2Mix, 16 kHz `min`, `mix_clean`).** Simulated from
LibriSpeech + WHAM! noise (both public). Generate and prepare (`--stage prep`
prints this):
```bash
git clone https://github.com/s3prl/LibriMix.git && cd LibriMix
# point LibriSpeech / WHAM! paths in generate_librimix_ss.sh at your copies:
./generate_librimix_ss.sh <WRITABLE_STORAGE_DIR>
cd <S3PRL>/s3prl
for part in train-100 dev test; do
  python3 downstream/separation_stft/scripts/LibriMix/data_prepare.py \
      --part $part --sample_rate 16k --mode min \
      <WRITABLE_STORAGE_DIR>/Libri2Mix downstream/separation_stft/datasets/Libri2Mix
done
python3 downstream/separation_stft/scripts/LibriMix/subsample.py \
    downstream/separation_stft/datasets/Libri2Mix/wav16k/min/dev \
    downstream/separation_stft/datasets/Libri2Mix/wav16k/min/dev_1000 --sample 1000
```
This yields `.../Libri2Mix/wav16k/min/{train-100,dev_1000,test}` (the default
`--data-root`; note the non-obvious subdir names). Metrics need
**`asteroid==0.4.4`** (see SE note). *Paper vs recipe:* SUPERB-SG describes SS
"simulated from LibriSpeech + WHAM! noise", but the shipped recipe separates
`mix_clean` (no noise); SI-SDRi is the comparable metric regardless.

**⚠ `subsample.py` hard-codes a 6-cond list (must handle for mix_clean-only
generation).** `downstream/separation_stft/scripts/LibriMix/subsample.py`
iterates a fixed `["mix_both","mix_clean","mix_single","noise","s1","s2"]` and
`os.makedirs`/reads `wav.scp` for each. When you generate **`mix_clean` only**
(the SUPERB-SG recipe), the `mix_both`/`mix_single` dirs never exist, so it
crashes on the first missing cond (the earlier `data_prepare.py` step already
correctly skips absent conds; only `subsample.py` assumes all six). **Fix
without touching s3prl source:** run a *patched copy* of `subsample.py` whose
cond list is `["mix_clean","noise","s1","s2"]`. Do **not** edit the clone. (A
crashed original run can also leave a stale, wav.scp-less `dev_1000/mix_both/`
behind — delete it; the loader reads only `src+tgt` = `mix_clean,s1,s2`, but a
half-written dir is confusing.)

**Run it.**
```bash
bash scripts/superb_sg/run_wavlm_ss.sh --exp-name wavlm_base_plus_ss
# optional: --data-root <dir-with-train-100/dev_1000/test>
```
Train: `-d separation_stft -c downstream/separation_stft/configs/cfg.yaml`.
Evaluate: `-m evaluate -t test -e result/downstream/<exp>/best-states-dev.ckpt`.
Stages: `all` / `prep` / `train` / `evaluate`.

**Reading the output.** Evaluate writes
`result/downstream/<exp>/test_metrics.txt` (`si_sdr` = SI-SDRi; `stoi`/`pesq`
also written). The wrapper prints
`RESULT superb_sg ss si_sdri=<val> (stoi=<> pesq=<>)`. (s3prl also prints
`Average si_sdr of N utts: <val>` to the eval log.)

**Expected resources (estimate).** ~1 day training; ~15–25 GB VRAM (frozen
Base+). total_steps=150000, batch=8, 2 speakers.

**Reference (SUPERB-SG).** WavLM was not in the SUPERB-SG paper and the WavLM
Base+ leaderboard row was **not fetchable from this environment** — do not invent
it. Ballpark from the paper's best (**HuBERT Large**): **SI-SDRi ≈ 10.45 dB**
(arXiv:2203.06849). Label any run "reference: leaderboard (not fetched)".

**Verification status.** Commands verified against s3prl code + benchmark papers
(R1 audit, 2026-07-17); dry-run tested; **SMOKE-executed end-to-end** on
locally-generated Libri2Mix (2026-07-19) — train → evaluate (SI-SDRi/STOI/PESQ).

### Executed runs
| Date | Config | Steps | Metric | Label | Reference |
|---|---|---|---|---|---|
| 2026-07-19 | WavLM Base+, frozen | 200 (of 150000) | **SI-SDRi 3.88 dB** (stoi 0.77, pesq 1.18) | SMOKE | ~10.45 dB (HuBERT-L ballpark; WavLM row not fetched) |

- Command (cluster): `bash scripts/superb_sg/run_wavlm_ss.sh --data-root <gen>/ss_prepared/Libri2Mix/wav16k/min --exp-name smoke_ss --extra-override "config.runner.total_steps=200,,config.runner.log_step=20,,config.runner.eval_step=100,,config.runner.save_step=100"`.
- Throughput: **~2.76 s/optimizer-step** on one H100 (frozen Base+, batch 8, grad_accum=1, full ~10 s utterances — no chunking, unlike SD). Test eval over 3000 utts took **~36 min** (PESQ/STOI dominate).
- **⚠ Full-run walltime — exceeds the 24 h cap.** At the measured throughput, the full 150000-step run projects to **~60–115 h (≈ 2.5–5 days)** depending on how much the smoke's frequent-eval overhead inflates the per-step time (full-run `eval_step=2000` vs smoke's 100). This does **not** fit one 24 h PBS job: run it as a **checkpoint-resumed chain** (config `save_step=10000`, `max_keep` high; resume from `result/downstream/<exp>/states-<step>.ckpt`), i.e. 3–5 sequential jobs. R1's "~1 day" estimate is optimistic vs the H100 smoke measurement; SS trains on full-length utterances, which is the cost driver.
- The 200-step SI-SDRi (3.88 dB) is a pipeline-sanity number, not a result. Data generated locally: Libri2Mix `wav16k/min` (mix_clean), 23 GB; train-100/dev_1000/test = 13900/1000/3000 mixtures.

## VC — Voice Conversion (any-to-one)

**What it measures.** VC converts speech from arbitrary source speakers to a
fixed target speaker's voice. This is the **any-to-one (a2o)** setting — the one
in the SUPERB-SG paper. On frozen WavLM features a Taco2-AR acoustic model is
trained **per target speaker** (TEF1, TEF2, TEM1, TEM2), a pretrained neural
vocoder synthesises the waveform, and quality is scored by **MCD** (mel-cepstral
distortion, dB; lower is better), **averaged over the four target speakers**.
(The any-to-any `a2a-vc-vctk` recipe is a separate ICASSP extension — see
`docs/superb_sg.md#extras`, not part of the benchmark.)

**Dataset.** VCC2020, **public**, plus a pretrained Parallel-WaveGAN vocoder:
```bash
cd downstream/a2o-vc-vcc2020/data && ./data_download.sh vcc2020/ && cd ../../..
# vocoders (PWG task1/task2 + HiFi-GAN) via gdown:
bash scripts/superb_sg/run_wavlm_vc_a2o.sh --stage vocoder   # -> downstream/a2o-vc-vcc2020/pwg_task1 ...
```
`data_download.sh` clones `nii-yamagishilab/VCC2020-database` into
`downstream/a2o-vc-vcc2020/data/vcc2020` (the default `--data-root`).
**Env note.** VC has its own extras
(`downstream/a2o-vc-vcc2020/requirements.txt`: `parallel-wavegan`, `fastdtw`,
`pyworld`, `pysptk`, `jiwer<4`, `resemblyzer`); `pyworld`/`pysptk`/`webrtcvad`
compile from source (need a C/C++ toolchain + Cython) and `resemblyzer`/
`parallel-wavegan` reference torch. Installed into the SUPERB env **without
touching the pinned `torch==2.4.1+cu121` / `numpy` / `torchaudio`** — pin those
in a pip constraints file and use two per-package workarounds:

```bash
# pin so nothing up/down-grades torch/torchaudio/numpy
printf 'torch==2.4.1+cu121\ntorchaudio==2.4.1+cu121\nnumpy==2.2.6\n' > /tmp/vc_constraints.txt

# 1) compiling + simple extras
pip install --constraint /tmp/vc_constraints.txt fastdtw pyworld pysptk "jiwer<4" webrtcvad

# 2) resemblyzer WITHOUT deps: its setup lists the `typing` PyPI backport, which
#    shadows/breaks stdlib typing on Python >=3.9. webrtcvad (its only otherwise-
#    missing runtime dep) is installed in step 1.
pip install --no-deps resemblyzer

# 3) parallel-wavegan needs --no-build-isolation: its sdist setup.py does
#    `import pip` at build time, which is unavailable inside pip's isolated build env.
pip install --no-build-isolation --constraint /tmp/vc_constraints.txt parallel-wavegan

# 4) restore pkg_resources for setuptools>=81 (it dropped the bundled module, which
#    pyworld/pysptk/resemblyzer/webrtcvad import at runtime).
pip install "setuptools<81"
```

Resolved versions: `parallel-wavegan 0.6.1, fastdtw 0.3.4, pyworld 0.3.5,
pysptk 1.0.1, jiwer 3.1.0, resemblyzer 0.1.4` (+ `webrtcvad, rapidfuzz, kaldiio`;
`setuptools 80.x`). torch/numpy/torchaudio untouched.

**scipy ≥ 1.13 (vocoder decode).** ParallelWaveGAN's `pqmf.py` does
`from scipy.signal import kaiser`, a top-level alias **removed in scipy 1.13**
(now `scipy.signal.windows.kaiser`); vocoder decoding `ImportError`s on newer
scipy (this env ships scipy 1.15). A VC-only env can just `pip install
"scipy<1.13"`. To keep a shared modern-scipy env, restore the alias at
interpreter startup via a `sitecustomize.py` on `sys.path`:

```python
import scipy.signal, scipy.signal.windows
if not hasattr(scipy.signal, "kaiser"):
    scipy.signal.kaiser = scipy.signal.windows.kaiser
```

**Vocoder download (gdown ≥ 5).** The recipe's `vocoder_download.sh` calls
`gdown --id <ID>`, a flag **removed in gdown ≥ 5** — it fails silently and leaves
empty vocoder dirs. The wrapper's `--stage vocoder` was patched to invoke gdown
with the current positional syntax (`gdown <ID> -O <tmp> && tar xzf ...`), so it
works regardless of the installed gdown version and fetches the same public
artifacts (`pwg_task1`, `pwg_task2`, `hifigan_vctk+vcc2020`). Only `pwg_task1` is
needed for the four task-1 English targets (TEF1/2, TEM1/2).

**Run it.**
```bash
# trains 4 target-speaker models then decodes + scores each (MCD averaged):
bash scripts/superb_sg/run_wavlm_vc_a2o.sh --data-root /path/to/vcc2020 \
     --exp-name wavlm_base_plus_vc
```
Options: `--trgspk TEF1|TEF2|TEM1|TEM2|all` (default `all`);
`--vocoder-dir` (default `downstream/a2o-vc-vcc2020/pwg_task1`); `--step`
(default `10000` = final checkpoint); `--stage all|vocoder|train|evaluate`.
There is **no dev-best checkpoint** (save_names=[]); the runner saves
`states-<step>.ckpt` every `save_step`. The test features at
`result/downstream/<exp>_<spk>/<step>/test/hdf5` that `decode.sh` vocodes are
written by the runner's **in-training evaluation** (its `eval_dataloaders`
include `test`), so `--step` must be a step at which that eval ran — i.e. a
multiple of `eval_step`; the default `10000` (eval_step `1000`) qualifies. The
wrapper's `evaluate` stage decodes that `<step>` dump **directly** — it does not
run a separate `-m evaluate` pass (a standalone `-m evaluate` writes to
`.../0/test/hdf5` with global_step 0, and its fresh WavLM re-extraction of the
test set can stall for hours on a contended node, so it is pure waste here).
(evaluate.py scores only files matching `*300??*`, the 25×4 eval utterances, so
the dev features co-dumped at each `eval_step` do not affect MCD.)

**Reading the output.** `decode.sh` prints
`Mean MCD, f0RMSE, f0CORR, DDUR, CER, WER, accept rate: <mcd> ...` (per speaker;
teed to `logs/superb_sg/vc_a2o/<exp>_<spk>_decode.log`, detailed per-utt in the
step dir's `obj.log`). The wrapper parses each MCD and prints
`RESULT superb_sg vc_a2o mcd=<avg> (mean over N speaker(s))`.

**Expected resources (estimate).** Fast to train: ~1–2 h per target speaker →
~4–8 h for the four + decoding; < 15 GB VRAM. total_steps=10000/spk, batch=6,
fbank fs=24000.

**Reference (SUPERB-SG).** WavLM was not in the SUPERB-SG paper and the WavLM
Base+ leaderboard row was **not fetchable from this environment** — do not invent
it. Ballpark from the paper's best (**HuBERT Large**): **MCD ≈ 7.22 dB**
(arXiv:2203.06849; reported with WER and ASV-acceptance side metrics). Label any
run "reference: leaderboard (not fetched)".

**Verification status.** Commands verified against s3prl code + benchmark papers
(R1 audit, 2026-07-17); dry-run tested; **SMOKE-executed end-to-end on VCC2020 in
this repo** (TEF1, 500 steps → in-training test dump → PWG vocoder decode →
objective eval → MCD; PBS job 193964, Exit 0, 2026-07-19).

### Executed runs

| Date | Run | Metric | Label |
|---|---|---|---|
| 2026-07-19 | TEF1, 500 steps (`--trgspk TEF1 --step 500`, override `total_steps=500,save_step=500,eval_step=500,log_step=100`) | MCD **11.35 dB** | **SMOKE** |

SMOKE is a pipeline proof only: with a 4000-step warmup a 500-step model is
essentially untrained, so MCD sits far above the ~7.22 dB reference (expected —
do not read as a benchmark). Full vocoder synthesis + objective eval ran over all
100 scored utterances.

**Measured stage costs** (500-step TEF1, one H100, under heavy node contention;
job walltime 7 min 28 s):
- **Train + one in-training eval** (dev+test feature dump): ~6 min — 500 steps at
  ~0.4 s/step steady-state, plus ~40 s for the step-500 dev+test dump.
- **`decode.sh`: ~77 s total** — normalize 110 hdf5 ~6 s; PWG synthesis of 110
  utts ~5 s (RTF 0.011); objective eval (Resemblyzer d-vector ASV +
  wav2vec2-large ASR + MCD/f0 over the 100 scored utts) ~65 s.

Decode is cheap and bounded; **training dominates**. The earlier walltime kill was
a now-removed redundant `-m evaluate` WavLM re-extraction, not decode.

**Full-run projection.** ~0.4 s/step ⇒ 10 000 steps ≈ **1–1.5 h/speaker**; the
default `eval_step=1000` adds 10 in-training evals ≈ ~7 min/speaker; decode ≈
~1–2 min/speaker. Four speakers + decode ≈ **~5–7 h sequential**, so a single
`--trgspk all` job at `walltime=12:00:00` has ~2× headroom. GPU memory is small
(Taco2-AR acoustic model; well under the ~15 GB the inventory estimated).

## ST — Speech Translation

**What it measures.** ST translates spoken English into German text end-to-end.
On frozen WavLM features a fairseq **`s2t_transformer`** encoder-decoder is
trained on CoVoST2 En→De and evaluated with **case-sensitive detokenized BLEU**
(sacreBLEU); higher is better.

**⚠ Requires fairseq.** The ST expert does `import fairseq` at module load, and
fairseq is **not** in `requirements/all.txt`. It is the highest infra risk here:
fairseq builds are fragile on torch 2.x / modern CUDA and often need a **pinned
fairseq in a dedicated env**. The wrapper **gates** on `python3 -c "import
fairseq"` and fails early with this pointer if it is missing. Also needs
`sacrebleu`, `sacremoses`, `sentencepiece` (the first and last are in `[all]`).

**Dataset — GATED.** CoVoST2 En→De = **Common Voice Corpus 4 (English)** audio +
the CoVoST2 translation tsvs. The tsvs are CC0 (fetched automatically), **but the
Common Voice v4 audio is gated**: it needs a Mozilla account and the *version-4*
archive from <https://commonvoice.mozilla.org/en/datasets>. (Locally we only have
Common Voice v21/v24 — the *wrong version*.) Prepare (`--stage prep` prints this):
```bash
# after obtaining CV4 English -> <CV4_EN_ROOT> (has en/clips/, en/validated.tsv):
# edit covo_root in downstream/speech_translation/prepare_data/prepare_covo.sh, then:
cd downstream/speech_translation/prepare_data/ && bash prepare_covo.sh
```
This builds tsv + SentencePiece vocab + `config.yaml` into `data/covost_en_de`
(the default `--data-root`).

**Run it.**
```bash
bash scripts/superb_sg/run_wavlm_st.sh --exp-name wavlm_base_plus_st
# optional: --data-root <prepared covost_en_de dir>
```
Train: `-d speech_translation` (default config; `s2t_transformer`, 3 enc / 3 dec
layers). Evaluate: `-m evaluate -t test -e result/downstream/<exp>/dev-best.ckpt`.
Stages: `all` / `prep` / `train` / `evaluate`.

**Reading the output.** The expert prints the sacreBLEU object
`BLEU = <score> <p1>/<p2>/... (BP ...)` to the eval log and writes
`result/downstream/<exp>/output-st-test.tsv`; the wrapper prints
`RESULT superb_sg st bleu=<score>`. Alternative scorer:
`python3 downstream/speech_translation/count_sacreBLEU.py --exp-dir
result/downstream/<exp> --tsv-file output-st-test.tsv`.

**Expected resources (estimate).** ~1–2 days training; ~20–30 GB VRAM (frozen
Base+; CoVoST En train ≈ 426 h of audio, transformer enc/dec). total_steps=32000,
grad_accum=8.

**Reference (SUPERB-SG).** WavLM was not in the SUPERB-SG paper and the WavLM
Base+ leaderboard row was **not fetchable from this environment** — do not invent
it. Ballpark from the paper's best (**HuBERT Large**): **BLEU ≈ 20.01**
(arXiv:2203.06849). Label any run "reference: leaderboard (not fetched)".

**Verification status.** Commands verified against s3prl code + benchmark papers
(R1 audit, 2026-07-17); dry-run tested; not yet executed on data in this repo.

## Optional extras (NOT part of the SUPERB / SUPERB-SG benchmarks)

These two recipes exist in s3prl and are adjacent to the tasks above, but they
are **not** scored tasks of the SUPERB-SG paper. No wrapper scripts are shipped
for them; run them by hand from `<S3PRL>/s3prl` if you want them. Any numbers are
extras, clearly separate from the 14-task benchmark.

### VC a2a — any-to-any voice conversion (S3PRL-VC, ICASSP 2022)

The `a2a-vc-vctk` recipe is the **any-to-any** extension from the S3PRL-VC paper
(arXiv:2110.06280), **not** the SUPERB-SG VC task (which is any-to-one,
`a2o-vc-vcc2020`). It trains a single speaker-independent model on **VCTK**
(`config_ar_taco2.yaml`, keys `downstream_expert.datarc.trdev_data_root` →
`.../VCTK-Corpus/wav48` and `eval_data_root` → `.../vcc2020`), conditions on
Resemblyzer d-vectors, and evaluates VCC2020 intra-lingual conversion with the
same **MCD** metric. How-to: get VCTK (Edinburgh DataShare `10283/3443`) + VCC2020
(`data_download.sh`), download a HiFi-GAN vocoder (`vocoder_download.sh`), then
`./downstream/a2a-vc-vctk/vc_train.sh` followed by
`./downstream/a2a-vc-vctk/decode.sh`. **Cost warning:** d-vector extraction alone
is ~5–6 h and VCTK is ~44 h of audio, so this is much heavier than a2o. Extra
deps mirror a2o (`downstream/a2a-vc-vctk/requirements.txt`: `parallel-wavegan`,
`fastdtw`, `pyworld`, `pysptk`, `jiwer<4`, `resemblyzer`).

### OOD-ASR — out-of-domain ASR (SUPERB Challenge extension)

`downstream/ctc/README.md` defines **out-of-domain ASR** on the same `ctc` recipe,
differing only by `-c` config: cross-lingual **Common Voice 7.0** Spanish / Chinese
/ Arabic via `downstream/ctc/cv_config/cv_{es,zh,ar}.yaml` (each sets a `cv_root`
key), plus spontaneous English **SBCSAE** via `downstream/ctc/sbcsae.yaml`. This
is the closest thing in this s3prl commit to the spec's "out-of-domain /
noise-robustness variants" — it is **ASR-only, config-driven**, not a roster of
noisy re-evals of every task. How-to (per language): download the data (Common
Voice 7.0 is **gated** — Mozilla account, and it is a *different version* from the
CoVoST2/ST v4 and from the local v21/v24), set `cv_root` (or the SBCSAE `path` +
`train/dev/test` tsvs) in the chosen config, then
`python3 run_downstream.py -m train -u wavlm_base_plus -d ctc -c
downstream/ctc/cv_config/cv_es.yaml -n wavlm_oodasr_es` and evaluate with
`-m evaluate -t test -e result/downstream/wavlm_oodasr_es/dev-best.ckpt`. Metric:
**WER / CER** (`downstream/ctc/metric.py`). Treat as optional; gated by Common
Voice 7.0 access.

**Verification status.** Recipe paths/keys verified against s3prl code (R1 audit,
2026-07-17); how-to not dry-run-scripted (no wrappers shipped for these extras).
