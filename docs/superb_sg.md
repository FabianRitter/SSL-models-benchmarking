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

**Dataset.** Voicebank-DEMAND, 16 kHz, **public**. Official source: Edinburgh
DataShare `10283/2791` (get the 16 kHz `clean_/noisy_ trainset_28spk_wav_16k`
and `testset_wav_16k` sets); s3prl also references a mirror
`http://140.112.21.28:9000/noisy-vctk-16k.zip` (old NTU host, may be dead).
Prepare the Kaldi-scp layout (`--stage prep` prints this):
```bash
for part in train dev test; do
  python3 downstream/enhancement_stft/scripts/Voicebank/data_prepare.py \
      <VB_SRC> downstream/enhancement_stft/datasets/voicebank --part $part --sample_rate 16k
done
```
This writes `downstream/enhancement_stft/datasets/voicebank/wav16k/{train,dev,test}`
(the default `--data-root`). **Env note:** metrics need **`asteroid==0.4.4`**
(pulls `pesq`, `pystoi`); that old pin is a known install-fragility point —
verify `python -c "from asteroid.metrics import get_metrics"` at env-build time.

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
(R1 audit, 2026-07-17); dry-run tested; not yet executed on data in this repo.

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
(R1 audit, 2026-07-17); dry-run tested; not yet executed on data in this repo.

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
**Env note:** VC has its own extras
(`downstream/a2o-vc-vcc2020/requirements.txt`: `parallel-wavegan`, `fastdtw`,
`pyworld`, `pysptk`, `jiwer<4`, `resemblyzer`); `pyworld`/`pysptk` need build
tools and `resemblyzer` pulls its own torch. The vocoder download uses `gdown`
(needs network).

**Run it.**
```bash
# trains 4 target-speaker models then decodes + scores each (MCD averaged):
bash scripts/superb_sg/run_wavlm_vc_a2o.sh --data-root /path/to/vcc2020 \
     --exp-name wavlm_base_plus_vc
```
Options: `--trgspk TEF1|TEF2|TEM1|TEM2|all` (default `all`);
`--vocoder-dir` (default `downstream/a2o-vc-vcc2020/pwg_task1`); `--step`
(default `10000` = final checkpoint's feature dump); `--stage
all|vocoder|train|evaluate`. There is **no dev-best checkpoint** (save_names=[]);
evaluate re-dumps test features from `states-<step>.ckpt` into
`result/downstream/<exp>_<spk>/<step>/test/hdf5`, which `decode.sh` vocodes and
scores.

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
(R1 audit, 2026-07-17); dry-run tested; not yet executed on data in this repo.

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
