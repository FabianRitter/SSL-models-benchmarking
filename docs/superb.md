# SUPERB — task-by-task guide (WavLM Base+ worked example)

The ten SUPERB tasks (Interspeech 2021), each runnable end-to-end with one wrapper
script from the repo root. All wrappers share the same flags (`--data-root`,
`--upstream`, `--gpu`, `--exp-name`, `--stage`, `--extra-override`, `--dry-run`) and
assume the conda environment from the [README install](../README.md#install).
Reference numbers are WavLM Base+ from the WavLM paper (arXiv:2110.13900, Table I).

**Verification labels used below:** *SMOKE-verified* = the script was executed
end-to-end on real data with reduced steps (proves the pipeline; the printed number
is NOT a benchmark result). *Dry-run verified* = command plan validated, not yet
executed on data here. Full benchmark runs use each script's DEFAULT settings.

## PR — Phoneme Recognition

**What it measures.** PR trains a lightweight CTC head (a learnable weighted sum
over the frozen WavLM hidden states feeding a BLSTM projection) to transcribe
speech into phoneme sequences, and reports the **phoneme error rate (PER)** on
LibriSpeech `test-clean`. Because the upstream is frozen and only the
weighted-sum + small downstream head is trained, PER is a direct probe of how
much phonetic content the self-supervised representation exposes. Lower is better.

**Dataset.** LibriSpeech `train-clean-100` (train), `dev-clean` (validation),
`test-clean` (test). Public, no gating: download the three tarballs from
<https://www.openslr.org/12> (`train-clean-100.tar.gz`, `dev-clean.tar.gz`,
`test-clean.tar.gz`) and extract them under one directory so it contains
`train-clean-100/`, `dev-clean/`, `test-clean/` (~7 GB for the three splits).
That directory is `--data-root`. The phoneme vocabulary and the g2p lexicons ship
in-repo (`downstream/ctc/vocab/phoneme.txt`, `downstream/ctc/lexicon/`), so no
grapheme-to-phoneme step is needed.

**Run it.**
```bash
bash scripts/superb/run_wavlm_pr.sh --data-root /path/to/LibriSpeech --exp-name wavlm_base_plus_pr_full
```
This trains (`-d ctc -c downstream/ctc/libriphone.yaml`) then evaluates on
`test-clean` and prints the metric. The `ctc` recipe is shared by PR / SF /
OOD-ASR and is specialised to PR purely by `libriphone.yaml`. Options:
`--upstream` (default `wavlm_base_plus`), `--gpu`, `--stage all|train|evaluate`,
`--extra-override` (verbatim s3prl `-o` string, `,,`-separated), `--dry-run`.

**Reading the output.** The evaluate stage prints `test per: <fraction>`
(s3prl `downstream/ctc/expert.py`); the wrapper greps the last such line and
prints a final line `RESULT superb pr per=<fraction>`. Multiply by 100 for PER%.
Artifacts land in `result/downstream/<exp-name>/` inside the s3prl checkout
(`dev-best.ckpt`, `config_*.yaml`, `{dev,test}-hyp.ark` / `-ref.ark`); wrapper
logs go to `logs/superb/pr/<exp-name>_{train,eval}.log`.

**Expected resources.** Config defaults: `total_steps=100000`, `batch_size=16`,
`gradient_accumulate_steps=2`, Adam `lr=1e-2` (`downstream/ctc/libriphone.yaml`).
Full run ≈ 10–16 h on a single 80 GB GPU at < 15 GB VRAM (estimate). Smoke-
measured throughput: ~0.76 s per optimiser step (~1.3 it/s steady-state training)
on one H100 while shared with 4 concurrent sibling smokes; the periodic full-dev
evaluations add to this, and a dedicated GPU is materially faster. Recommended
PBS walltime for the full run: **36:00:00** (comfortable headroom over a shared
GPU).

**Reference (WavLM Base+).** PER **3.92 %** (WavLM paper, arXiv:2110.13900,
Table I). WavLM Base = 4.84 %, WavLM Large = 3.06 %.

### Executed runs

| Date (UTC+8) | Upstream | Command (shrink override) | Result | Label |
|---|---|---|---|---|
| 2026-07-17 | wavlm_base_plus | `run_wavlm_pr.sh --exp-name smoke_pr --extra-override "config.runner.total_steps=200,,config.runner.eval_step=100,,config.runner.save_step=100,,config.runner.log_step=50,,config.runner.evaluate_ratio=0.1,,config.downstream_expert.corpus.num_workers=8"` | test per=0.0835 (**8.35 %**) | **SMOKE** |

> SMOKE = 200 training steps + evaluation on 10 % of `test-clean`
> (`evaluate_ratio=0.1`); a pipeline sanity check only, **not** a benchmark
> number (the full recipe is 100 000 steps on the whole test set). It proves the
> wrapper runs train → `dev-best.ckpt` → evaluate → the `RESULT superb pr per=`
> line end-to-end. Wall time 8m9s on a contended H100. `evaluate_ratio` and the
> reduced `num_workers` are smoke-only knobs; the full-run command above sets
> none of them (full test set, config-default workers).

## ASR — Automatic Speech Recognition

**What it measures.** Frame-level acoustic modelling: the frozen WavLM features are
weighted-summed (learnable) and fed to a 2-layer BiLSTM + CTC head that transcribes speech
to characters. The reported metric is **Word Error Rate (WER, lower is better)** on
LibriSpeech `test-clean`, decoded **without any language model** (greedy CTC). This is the
"no-LM" ASR row of the WavLM/SUPERB tables; adding a 4-gram/Transformer LM would lower WER by
a couple of points but is intentionally out of scope here.

**Dataset.** [LibriSpeech](https://www.openslr.org/12/) — public, no gating. The recipe uses
`train-clean-100` (training), `dev-clean` (checkpoint selection) and `test-clean` (scoring).
Download `train-clean-100.tar.gz`, `dev-clean.tar.gz` and `test-clean.tar.gz` from openslr.org/12
and extract them so the root contains `train-clean-100/`, `dev-clean/`, `test-clean/`. Point
`--data-root` at that root (its path must contain the string `LibriSpeech`, which the standard
archive layout already satisfies).

**Run it (one command):**

```bash
bash scripts/superb/run_wavlm_asr.sh --data-root /path/to/LibriSpeech \
     --exp-name wavlm_base_plus_asr_full
```

`--stage all` (default) runs three sub-steps in order:
1. **prep** — one-off `preprocess/generate_len_for_bucket.py` over `train-clean-100`,
   `dev-clean`, `test-clean`, producing the length-bucket table
   `data/librispeech/len_for_bucket/*.csv` that the dataloader buckets on. Auto-skipped if the
   three CSVs already exist. CPU-only, ~1–2 min with `PREP_NJOBS=12`.
2. **train** — `run_downstream.py -m train -u wavlm_base_plus -d asr` (frozen upstream,
   learnable weighted-sum; SpecAugment on; Adam lr 1e-4; 200 000 steps; batch 32).
3. **evaluate** — scores `dev-clean-best.ckpt` on `test-clean` (**no LM**) and prints the WER.

Re-run a single phase with `--stage prep|train|evaluate`. Other flags: `--upstream`, `--gpu`,
`--extra-override` (verbatim s3prl `-o`, `,,`-separated), `--dry-run`.

**How to read the output.** The evaluate step prints (from `downstream/asr/expert.py`):

```
test-clean uer: <fraction>
test-clean wer: <fraction>
```

and the wrapper appends a final machine-readable line:

```
RESULT superb asr wer=<percent>       # e.g. 5.59  ==  5.59 %
```

WER is emitted **as a percentage** (0–100, computed as `100 * word_errors / words` in
`downstream/asr/expert.py:_compute_metrics`) — report it directly, no scaling. Artifacts land in
`external/s3prl/s3prl/result/downstream/<exp-name>/` (checkpoints, `test-clean-noLM-hyp.ark` /
`-ref.ark`, tensorboard). Wrapper logs are under `logs/superb/asr/<exp-name>_{prep,train,eval}.log`.
Note the selection checkpoint is **`dev-clean-best.ckpt`** (the runner selects on `dev-clean`),
not `dev-best.ckpt`.

**Expected resources (1×H100).** ~20–30 GB VRAM. Training throughput measured at **~0.2 s/step**
on a lightly-loaded H100 and **~0.45 s/step** under contention from sibling jobs (frozen Base+,
batch 32). The full 200 000-step run has two comparable cost centres: training itself (~12–25 h)
and the periodic **full** `dev-clean` evaluations (default `eval_step=2000` ⇒ 100 evals × ~2 700
utts at `eval_batch_size=1`, another ~8–30 h depending on contention). Budget **~1.5–3 days**
wall-clock end-to-end. Suggested scheduler walltime: **96 h**.

**Reference (WavLM paper, Table I — arXiv:2110.13900).** WavLM Base+ ASR **WER 5.59 %** (no LM)
on `test-clean` — i.e. the wrapper prints `RESULT superb asr wer=5.59`. (Base 6.21 %, Large
3.44 %.) The public SUPERB leaderboard may differ by small amounts.

### Executed runs

| Date | Command (abridged) | Steps | Metric | Label |
|---|---|---|---|---|
| 2026-07-17 | `run_wavlm_asr.sh --stage all --extra-override total_steps=1000,,eval_step=500,,save_step=500,,log_step=100,,evaluate_ratio=0.1` | 1000 | `wer=27.66` (test-clean, 262-utt subset, no LM) | **SMOKE** |

> **SMOKE** = pipeline sanity only (1 000 steps; `evaluate_ratio=0.1` scores ~10 % of each split
> to keep the sanity run short). This 27.66 % WER is **not** a benchmark result — it only confirms
> the head is learning (dev-clean WER fell 40.3 % → 27.2 % across the two in-training evals) and
> that prep → train → `dev-clean-best.ckpt` → evaluate → `RESULT` runs end-to-end. Measured ~5 min
> wall on 1×H100. The real number goes in `results/RESULTS.md` after the 200 000-step run.
>
> **Gotcha for tiny smokes:** `downstream/asr/expert.py` initialises `best_score = 100` and saves
> `dev-clean-best.ckpt` only when dev WER `< 100`. A ≤100-step model outputs *exactly* 100.0 % WER,
> so the checkpoint is never written and the evaluate stage has nothing to score. Use **≥ ~500
> steps** (and enough that dev WER drops below 100) for a self-checking smoke.

## KS — Keyword Spotting

**What it measures.** KS asks the model to classify a 1-second utterance into one of
**12 classes** — the ten Speech Commands keywords (*yes, no, up, down, left, right, on,
off, stop, go*) plus `_unknown_` and `_silence_`. The frozen upstream is turned into a
single utterance embedding by a learnable weighted-sum over its hidden layers followed
by mean-pooling; only a small linear projector + utterance classifier are trained. It is
a cheap, fast-converging probe of how linearly separable the upstream's phonetic/word
content is. **Metric: test accuracy (higher is better).**

**Dataset.** Google **Speech Commands v0.01** — **public, no gating**. Two tarballs:

| Tarball | URL | Extract into |
|---|---|---|
| Train + dev | `https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz` | `<data-root>/speech_commands_v0.01/` |
| Official test set | `https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz` | `<data-root>/speech_commands_test_set_v0.01/` |

Each tarball is extracted into **its own subdirectory** of a shared parent (`<data-root>`);
that parent is what you pass to `--data-root`. On-disk size is ~2.1 GB extracted
(~1.5 GB train + ~0.1 GB test set). The train/dev split is derived deterministically by a
hash of each speaker id (`downstream/speech_commands/dataset.py`), so no manual split is
needed. v0.02 works too — extract the v0.02 tarballs and point `--data-root` at them
(the wrapper expects the `..._v0.01` directory names; rename or adapt for v0.02).

**Run (train + evaluate, one command):**

```bash
bash scripts/superb/run_wavlm_ks.sh --data-root /path/to/speech_commands
# /path/to/speech_commands must contain BOTH:
#   speech_commands_v0.01/  and  speech_commands_test_set_v0.01/
```

Useful options: `--upstream <name>` (default `wavlm_base_plus`), `--exp-name <name>`,
`--stage all|train|evaluate`, `--gpu <ids>`, `--extra-override "<s3prl -o string>"`,
`--dry-run`. The upstream is **frozen** (weighted-sum over hidden states; no `-f`), matching
the published SUPERB protocol. The s3prl checkout is taken from `external/s3prl` by default
(override with the `S3PRL_ROOT` env var).

**How to read the output.** The evaluate stage prints, from
`downstream/speech_commands/expert.py`, a line `test acc: <fraction>` (e.g.
`test acc: 0.9737`). The wrapper parses it and prints the canonical last line:

```
RESULT superb ks acc=<fraction>
```

Multiply by 100 for the percentage the leaderboard/paper report. Artifacts land in
`external/s3prl/s3prl/result/downstream/<exp-name>/` (checkpoints `dev-best.ckpt`,
`states-*.ckpt`, per-split `*_predict.txt` / `*_truth.txt`, `log.log`); the wrapper's
own train/eval logs go to `logs/superb/ks/<exp-name>_{train,eval}.log`.

**Expected resources.** ~8–12 h on a single H100, **< 12 GB** VRAM (frozen Base+),
default `total_steps=200000`, `batch_size=32`, eval on dev+test every 5000 steps. The
smoke below measured steady-state training at **~15–21 steps/s (~0.05–0.07 s/step)** and
each in-training dev+test evaluation pass at ~10–13 s on an uncontended H100, so on this
hardware the full run may finish appreciably faster than the conservative 8–12 h estimate;
budget 24 h of walltime to be safe.

**Reference (WavLM paper, arXiv:2110.13900, Table I).** WavLM Base+ **97.37 %** accuracy
(WavLM Base 96.79, WavLM Large 97.86).

**Environment note (Python ≥ 3.10).** `downstream/speech_commands/expert.py` imports
`catalyst` at module load; the `catalyst==21.5` that s3prl's `requirements/all.txt` pulls
in references `collections.MutableMapping`, an alias **removed in Python 3.10**, so KS
training aborts on import with `AttributeError: module 'collections' has no attribute
'MutableMapping'`. Fix at the environment level (pick one): pin a Python-3.10-compatible
catalyst, or add a `sitecustomize.py` on the path that restores the aliases, e.g.
`import collections, collections.abc as a; collections.MutableMapping = a.MutableMapping`.
KS is the only SUPERB downstream that imports catalyst.

### Executed runs

| Date | Upstream | Command (key args) | Metric | Label |
|---|---|---|---|---|
| 2026-07-17 | wavlm_base_plus | `run_wavlm_ks.sh --data-root <ks-root> --exp-name smoke_ks --extra-override "config.runner.total_steps=300,,config.runner.eval_step=100,,config.runner.save_step=100,,config.runner.log_step=50"` | `test acc = 0.8539` (85.39 %) | **SMOKE** |
| 2026-07-19 | wavlm_base_plus | `run_wavlm_ks.sh --data-root <ks-root> --exp-name wavlm_base_plus_ks_full` (default = paper-faithful config) | `test acc = 0.9688` (**96.88 %**) | **FULL** |

> The 85.39 % above is a **300-step pipeline-sanity run only** — it verifies train→eval→
> `RESULT` end-to-end and is **not comparable** to the 97.37 % reference (a full 200k-step
> run). It nonetheless confirms the probe learns (train acc rose 0.23→0.86 over 300 steps).
> Run on 1×H100, end-to-end wall time 72 s.

## QbE — Query-by-Example Spoken Term Detection

**What it measures.** QbE searches for spoken query terms inside spoken
documents **without any training**: frozen WavLM features are extracted and
Dynamic Time Warping (DTW) aligns each query against each document. Following
the SUPERB protocol, DTW is run **layer-by-layer**, the best layer is chosen on
the dev set, and that layer's score is reported on the test set. The metric is
**MTWV** (maximum term weighted value); higher is better. It is a content probe
requiring no downstream weights.

**Dataset.** QUESST14 (`quesst14Database`), ~1.2 GB download, **public, no
gating**:
```bash
wget -c https://speech.fit.vutbr.cz/files/quesst14Database.tgz
tar zxf quesst14Database.tgz -C <CORPORA_DIR>
```
The extracted `quesst14Database/` directory is `--data-root`; it ships the
`scoring/` NIST toolkit (`score-TWV-Cnxe.sh`, `MediaEvalQUESST2014.jar`) and the
`groundtruth_quesst14_dev` / `groundtruth_quesst14_eval` reference dirs plus the
`language_key_{dev,eval,utterances}.lst` lists the scorer and dataloader need.
Requires `perl`, `java` (JRE), and `bc` on the compute node (the scorer also
tries `gnuplot`+`ps2pdf` for DET/TWV PDFs, but only *after* the MTWV line is
printed, so those are optional). In the env: `dtw-python==1.3.0` and `lxml`
(both in the s3prl `[all]` extras) — see the dtw-python caveat below.

> **Scope note (English-only).** The s3prl `quesst14_dtw` recipe evaluates the
> **English subset only** (`dataset.py` keeps `nnenglish` entries). That is
> **138 dev queries / 138 eval queries × 2438 documents**, i.e. ~336k DTW
> alignments per layer per split — roughly **20× smaller** than the full
> multilingual QUESST14 (560 queries × 12492 docs) that a naive estimate
> assumes. Runtime is correspondingly much lighter (see resources).

**Run it.**
```bash
# Full layer sweep (dev + test) then scoring; picks the best-dev layer:
bash scripts/superb/run_wavlm_qbe.sh --data-root /path/to/quesst14Database \
     --exp-name wavlm_base_plus_qbe
```
Options: `--layer all` (default; sweeps `0..num-layers-1`) or a single index;
`--num-layers` (default **13** = WavLM Base/Base+; use **25** for WavLM Large);
`--dist-fn` (default `cosine`; the config default is `cosine_exp`, which the
s3prl docs note is often best — try both); `--split dev|test|both` (default
`both` = the benchmark protocol; `dev` runs/scores only the dev queries, useful
for a single-layer smoke); `--stage all|dtw|score`; `--gpu`; `--extra-override`;
`--dry-run`. The recipe is `-d quesst14_dtw` with no `-c` and no checkpoint —
each run writes `benchmark.stdlist.xml` and the shipped scorer turns it into
TWV/MTWV/Cnxe. Bound the DTW pool to the allocated cores with
`--extra-override config.downstream_expert.max_workers=<ncpus>` (the config
default is empty → `os.cpu_count()`, which oversubscribes under a cgroup cpu
limit).

**Reading the output.** For each layer `L` and split `SP`, the DTW stage writes
`result/downstream/<exp>_L<L>_<SP>/benchmark.stdlist.xml` inside the s3prl
checkout; the score stage runs `score-TWV-Cnxe.sh` from `<data-root>/scoring`
and tees each scorer log to `logs/superb/qbe/<exp>_L<L>_<SP>_score.log`. The
scorer prints its result as `actTWV: <v>  maxTWV: <v>  Threshold: <t>` — the
**`maxTWV`** value IS the MTWV. The wrapper parses each layer's MTWV, selects
the best layer by **dev** MTWV, and prints
`RESULT superb qbe mtwv=<test-MTWV of best-dev layer> (test; best-dev layer=<L>)`.
(With `--split dev` it reports the best-dev layer's dev MTWV instead.)

**Expected resources.** No GPU training; GPU is used only for feature
extraction (~2576 short English audio files through WavLM Base+ — fast). The DTW
sweep is **CPU-bound**; with the English-only subset each layer is ~336k
alignments per split, parallelised across the allocated cores. Peak GPU VRAM is
small (feature extraction only). Measured per-layer wall time on 14 cores: see
Executed runs.

**Reference (WavLM Base+).** MTWV **0.0988** (WavLM paper, arXiv:2110.13900,
Table I). WavLM Base = 0.0870, WavLM Large = 0.0886. Higher is better.

**Full-run procedure (benchmark protocol).** Sweep every hidden layer on **both**
splits, pick the layer with the highest **dev** MTWV, report that layer's
**test** MTWV:
```bash
bash scripts/superb/run_wavlm_qbe.sh --data-root /path/to/quesst14Database \
     --layer all --split both --exp-name wavlm_base_plus_qbe \
     --extra-override config.downstream_expert.max_workers=14
```
Because the recipe is English-only, the whole 13-layer × 2-split sweep is
tractable in a **single** GPU job (est. from the smoke's per-layer time; the
wrapper loops all layers internally). To shorten wall-clock, split the layer
range across a few jobs (e.g. `--layer 0..6` and `--layer 7..12` via separate
invocations of a single `--layer <i>`), each doing `--split both`, then compare
dev MTWV across all per-layer score logs to pick the reported test layer.

### Executed runs
| Date | Split/Layer | Command (abridged) | MTWV | Wall (14 cores) | Label |
|---|---|---|---|---|---|
| 2026-07-19 | dev, layer 6 | `run_wavlm_qbe.sh --layer 6 --split dev --dist-fn cosine --extra-override ...max_workers=14` | _pending (PBS job 193842 queued)_ | _pending_ | SMOKE |

**Verification status.** Commands verified against s3prl code + benchmark papers
(R1 audit, 2026-07-17). 2026-07-19 (Smoke-Runner): wrapper extended with a
generic `--split dev|test|both` selector so a single-layer dev-only smoke is
expressible; `grep_mtwv` fixed to match the real scorer output (`maxTWV:`) and
hardened against `set -e`. The shipped NIST scorer (`score-TWV-Cnxe.sh` +
`MediaEvalQUESST2014.jar`) was run end-to-end on the dataset's example system,
confirming the MTWV line format and that `java`/`bc` suffice (`gnuplot`/`ps2pdf`
absence is non-fatal). **Env fix:** `dtw-python==1.3.0`'s PyPI manylinux wheel is
compiled against numpy 1.x and fails to import under the env's numpy 2.2.6
(`ValueError: numpy.dtype size changed`); rebuilt from source in
`envs/ssl-bench-s3prl` (`pip install --force-reinstall --no-deps
--no-build-isolation --no-binary dtw-python dtw-python==1.3.0`) so
`from dtw import dtw` works — required for QbE. Dry-run verified; smoke submitted
as PBS job 193842 (dev, layer 6), MTWV pending queue.

## SID — Speaker Identification

**What it measures.** Whether the frozen WavLM Base+ representation linearly separates
**speaker identity**. VoxCeleb1 is treated as a **closed-set 1251-way classification** task:
the downstream head (a learnable weighted-sum over WavLM's hidden states → mean-pooled
utterance vector → 256-d projector → linear classifier over the 1251 speakers) predicts which
of the 1251 celebrities produced each utterance. The metric is **accuracy** over the test
utterances (`downstream/voxceleb1/expert.py:124`). Note the split is **by utterance, not by
speaker**: all 1251 speakers appear in train, dev and test — the assignment is read from the
index column (1/2/3) of the shipped `downstream/voxceleb1/veri_test_class.txt`.

**Dataset — VoxCeleb1 dev+test wav, ~34 GB extracted** (1251 speaker dirs, 153 516 wavs; 16 kHz
mono). *Access is agreement-gated:* the official release is behind the VGG / University of Oxford
data-agreement page (robots.ox.ac.uk/~vgg/data/voxceleb). The practical public mirror used here is
the Hugging Face dataset **`ProgramComputer/voxceleb`**, which ships the dev set as four split parts
plus the test zip:
```bash
# 1. Fetch the four dev parts + the test zip from the HF mirror
huggingface-cli download ProgramComputer/voxceleb --repo-type dataset \
    --include "vox1/vox1_dev_wav_part*" "vox1/vox1_test_wav.zip" --local-dir ./vox1_dl
# 2. Concatenate the dev parts and unzip BOTH archives into a single merged wav/ tree
cd vox1_dl/vox1 && cat vox1_dev_wav_part* > vox1_dev_wav.zip
unzip -q vox1_dev_wav.zip     # -> wav/idXXXXX/<video>/<utt>.wav   (dev speakers)
unzip -q vox1_test_wav.zip    # -> wav/idXXXXX/<video>/<utt>.wav   (merges into the same wav/)
```

**Required layout (read carefully — this is the one non-obvious step).** s3prl locates each wav by
globbing **`<data-root>/*/wav/<speaker>/<video>/<utt>.wav`** (`downstream/voxceleb1/dataset.py`). The
`*` matches **one** intermediate directory and is a pure wildcard — it is **not** split-aware (dev vs
test is decided by `veri_test_class.txt`, never by the folder). So the correct `--data-root` is a
directory holding **one** sub-directory that contains the single merged `wav/` tree:
```
<data-root>/
  voxceleb1/            # any single name; matched by the "*"
    wav/
      id10001/<video>/00001.wav
      ...  (all 1251 speakers, dev+test merged)
```
Build it by `mv wav <data-root>/voxceleb1/wav` (or symlink `<data-root>/voxceleb1/wav -> .../wav`).
Do **not** create both `dev/wav` and `test/wav` pointing at the full tree — that makes the glob match
every utterance twice; a single merged tree is both correct and what the by-utterance split expects.

**Run (one command).**
```bash
bash scripts/superb/run_wavlm_sid.sh \
  --data-root /path/to/VoxCeleb1 \
  --exp-name wavlm_base_plus_sid_full
```
Trains (frozen upstream, learnable weighted-sum — no `-f`/`-l`, per SUPERB) then evaluates the test
split in a single invocation. `--stage train|evaluate` runs one half; `--dry-run` prints the exact
s3prl commands without executing; `--extra-override` appends verbatim `,,`-separated s3prl overrides
(used for the smoke below). The default config is `downstream/voxceleb1/config.yaml` (total_steps
200000, Adam lr 1e-4, gradient_accumulate_steps 4, train_batch 8, eval_batch 1, `max_timestep`
128000 = 8 s cap, eval every 5000 steps on dev+test, save every 1000). The wrapper injects only the
data-root override `config.downstream_expert.datarc.file_path=<data-root>`.

> **First-run cache.** On the first invocation the dataset scans `veri_test_class.txt` and globs all
> ~153 k utterances, caching the file lists to `downstream/voxceleb1/.cache/{train,dev,test}.pkl`.
> This is a one-off, CPU-bound step measured at **~2.8 min** here (train 138 361 / dev 6904 / test
> 8251 files); every later run loads the pickles instantly.

**Expected resources (1×H100).** Full run is 200 000 steps. Measured training throughput on an H100
**shared with five concurrent sibling smokes** was **~0.85 s/step** (effective batch 32; the tqdm
`overall` bar shows ~1.9 s/it because the in-training dev-eval round is folded into the final step).
That projects to **~47 h training + ~2.6 h of periodic dev+test evaluation ≈ ~50 h under contention**;
an **idle/dedicated** node should land around **~30–35 h**. Eval throughput was **~65–68 utt/s**
(batch 1). VRAM stays well under 80 GB (R1 estimate ~20 GB). **Budget PBS walltime `48:00:00`** — safe
for a dedicated node; do **not** launch the full job on a contended node, where it would exceed 48 h.

**Reading the output.** The evaluate stage prints `test acc: <fraction>` (from
`downstream/voxceleb1/expert.py:124`); the wrapper parses the last occurrence and prints the canonical
line
```
RESULT superb sid acc=<fraction>
```
Multiply by 100 for the leaderboard percentage. Checkpoints, config/args snapshots, per-utterance
prediction/truth files and tensorboard events land in `result/downstream/<exp-name>/`; the evaluate
stage loads `dev-best.ckpt`.

**Reference (paper).** WavLM Base+ scores **SID acc = 89.42 %** in the WavLM paper (arXiv:2110.13900,
Table I). For context: Base 84.51 %, Large 95.49 %.

### Executed runs
| Date | Command (abridged) | Metric | Label |
|---|---|---|---|
| 2026-07-17 | `run_wavlm_sid.sh --exp-name smoke_sid --extra-override "config.runner.total_steps=100,,eval_step=100,,save_step=100,,log_step=20,,eval_dataloaders=[\"dev\"]"` | `acc=0.001697` (0.17 %) | **SMOKE** |

SMOKE = a 100-step sanity run (vs. 200 000 for the full config) that proves the wrapper builds the
layout, trains → evaluates → prints `RESULT superb sid acc=` end-to-end. The 0.17 % accuracy is a
barely-trained pipeline check (chance ≈ 1/1251 = 0.08 %), **not** a benchmark number. The smoke
restricts in-training eval to `["dev"]` to save time; the final evaluate stage still scores the full
test split. Wall time was **5.5 min** (train ~1.5 min + dev eval ~1.8 min + test eval ~2 min), on an
H100 shared with sibling smoke jobs; the ~2.8 min path-cache build was done once beforehand.

## ASV — Automatic Speaker Verification

**What it measures.** ASV decides whether two utterances come from the same
speaker. On frozen WavLM features (learnable weighted-sum) the recipe trains an
X-Vector encoder with an AM-Softmax objective, then scores the VoxCeleb1 test
trials by cosine similarity. The metric is **EER** (equal error rate, %); lower
is better. Unlike most SUPERB tasks there is **no dev-best checkpoint**: the
recipe saves `states-<step>.ckpt` and every checkpoint is scored, reporting the
best EER.

**Dataset.** VoxCeleb1 (same audio as SID), dev + test wav, ~33 GB. **Not
license-gated but account/agreement-based and over the repo's staging budget**;
absent on this cluster. Official page:
<https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html>; practical mirror
`HuggingFace: ProgramComputer/voxceleb` (the exact `wget` is in s3prl's
`downstream/docs/superb.md`). Lay the audio out as
`<root>/wav/id1XXXX/<video>/<utt>.wav` — the shipped trial list
`downstream/sv_voxceleb1/voxceleb1_test_v2.txt` refers to that tree. That
directory is `--data-root`. On the first run the recipe filters utterances
< 2 s and caches paths under `downstream/sv_voxceleb1/cache_wav_paths`.

**Run it.**
```bash
bash scripts/superb/run_wavlm_asv.sh --data-root /path/to/VoxCeleb1 \
     --exp-name wavlm_base_plus_asv
```
Train stage: `-d sv_voxceleb1` (default config, frozen upstream, no `-f`/`-l`).
Evaluate stage: the shipped `downstream/sv_voxceleb1/test_expdir.sh
result/downstream/<exp> <VoxCeleb1_root>` loops `states-{20000..200000}.ckpt`,
runs `-m evaluate` on each, greps `test-EER`, and prints the best. Options:
`--stage all|train|evaluate`, `--upstream`, `--gpu`, `--extra-override`,
`--dry-run`.

**Reading the output.** Per-checkpoint EERs land in
`result/downstream/<exp>/report.txt`; `test_expdir.sh` prints
`The best checkpoint achieves EER <val>` and the wrapper echoes
`RESULT superb asv eer=<val>`. EER is a **fraction** from
`sv_voxceleb1/expert.py` (printed as `sv-voxceleb1/test-EER: <val>`); ×100 = EER %.

**Expected resources (estimate).** ~1.5–2 days training + several hours for the
checkpoint eval loop; ~20 GB VRAM (frozen Base+; long VoxCeleb utterances,
`max_timestep=128000` ≈ 8 s cap). total_steps=200000, save_step=10000, up to 20
checkpoints. Scoring every checkpoint is the slow part — parallelise on a second
GPU if available (the s3prl docs do exactly this).

**Reference (WavLM Base+).** EER **4.07 %** (WavLM paper, arXiv:2110.13900,
Table I). WavLM Base = 4.69 %, WavLM Large = 3.77 %. Lower is better.

**Verification status.** Commands verified against s3prl code + benchmark papers
(R1 audit, 2026-07-17); dry-run tested; not yet executed on data in this repo.

## SD — Speaker Diarization

**What it measures.** SD labels *who speaks when* in a 2-speaker mixture. On
frozen WavLM features (learnable weighted-sum) the recipe trains a small
RNN that emits per-frame per-speaker activity with a permutation-invariant
loss, infers RTTM segmentations, and scores **DER** (diarization error rate, %)
with `dscore`; lower is better. The checkpoint used for inference is
`best-states-dev.ckpt` (**not** `dev-best.ckpt`).

**Dataset — GENERATE (Libri2Mix).** No external data-root: the data are
**simulated** from LibriSpeech + WHAM! noise. Both ingredients are public
(LibriSpeech openslr.org/12; WHAM! noise wham.whisper.ai). Generate, then
convert to the diarization layout (`bash scripts/superb/run_wavlm_sd.sh --stage
prep` prints this):
```bash
git clone https://github.com/s3prl/LibriMix.git && cd LibriMix
# point the LibriSpeech / WHAM! paths in generate_librimix_sd.sh at your copies:
bash generate_librimix_sd.sh <WRITABLE_STORAGE_DIR>
python3 scripts/prepare_diarization.py \
    --target_dir <S3PRL>/s3prl/downstream/diarization/data \
    --source_dir <WRITABLE_STORAGE_DIR>/Libri2Mix/wav16k/max/metadata
```
This writes `downstream/diarization/data/{train,dev,test}` (the default
`--data-root`). Scoring additionally needs a clone of
`https://github.com/ftshijt/dscore` (deps: `intervaltree`, …).

**⚠ dscore path (must handle).** The shipped `downstream/diarization/score.sh`
**hardcodes** `dscore_dir=/groups/leo1994122701/dscore` (line 23). Rather than
edit the s3prl clone, the wrapper writes a **patched copy** of `score.sh` into
`logs/superb/sd/` with your `--dscore-dir` substituted and runs that. So pass
`--dscore-dir /path/to/dscore`. (Manual alternative: edit that one line in your
own copy of `score.sh`.)

**Run it.**
```bash
bash scripts/superb/run_wavlm_sd.sh --dscore-dir /path/to/dscore \
     --exp-name wavlm_base_plus_sd
# optional: --data-root <dir-with-train/dev/test>  --frame-shift 160 (shortens seqs)
```
Stages: `all` (= train + infer + score), or `prep` / `train` / `infer` / `score`.
`--data-root` defaults to `downstream/diarization/data`.

**Reading the output.** Inference writes RTTM predictions as `*.h5` under
`result/downstream/<exp>/scoring/predictions/`. The score stage sweeps
median∈{1,11} × threshold∈{0.3..0.7}, runs dscore, and the wrapper reports the
**lowest DER** across the sweep: `RESULT superb sd der=<val> (lowest ...)`. DER
is column 4 of the `*** OVERALL ***` dscore line (the `grep` filename prefix
shifts it, which is why `score.sh` sorts `-nrk 4`); the sorted table is in
`logs/superb/sd/<exp>_score.log`.

**Expected resources (estimate).** ~6–12 h training + scoring; < 15 GB VRAM
(frozen Base+). total_steps=30000, batch=8, grad_accum=4, 2 speakers, chunk 2000.
Data generation (Libri2Mix `max`) itself produces tens of GB and takes a while.

**Reference (WavLM Base+).** DER **3.50 %** (WavLM paper, arXiv:2110.13900,
Table I). WavLM Base = 4.55 %, WavLM Large = 3.24 %. Lower is better.

**Verification status.** Commands verified against s3prl code + benchmark papers
(R1 audit, 2026-07-17); dry-run tested; not yet executed on data in this repo.

## IC — Intent Classification

**What it measures.** Whether the frozen WavLM Base+ representation carries enough
utterance-level semantics to recognise a spoken command's *intent*. Each Fluent Speech
Commands utterance is labelled with three slots — **action**, **object**, **location** —
and the downstream head (a learnable weighted-sum over WavLM's hidden states → mean-pooled
utterance vector → per-slot linear classifier) predicts all three. The metric is
**accuracy**, but a prediction is correct only when **all three slots match** (exact-match
over the joint intent), computed in `downstream/fluent_commands/expert.py`.

**Dataset — Fluent Speech Commands (FSC), ~2.2 GB.** ~30 k single-channel 16 kHz wavs
(train 23 132 / valid 3 118 / test 3 793). The data root must **directly** contain `wavs/`
and `data/{train,valid,test}_data.csv`; the classic mistake is a double-nested directory,
so the wrapper checks that `data/train_data.csv` exists and aborts with a clear message if
not. *Access is license-gated:* the original is distributed by Fluent.ai behind a sign-up
form, and s3prl's own `downstream/docs/superb.md` points to a Hugging Face backup
`leo19941227/fluent_speech_commands` (`fluent.tar.gz`). Both the tarball and the local copy
bundle the **Fluent Speech Commands Public License** PDF — read it before use or redistribution.

**Run (one command).**
```bash
bash scripts/superb/run_wavlm_ic.sh \
  --data-root /path/to/fluent_speech_commands_dataset \
  --exp-name wavlm_base_plus_ic_full
```
Trains (frozen upstream, learnable weighted-sum — no `-f`/`-l`, per SUPERB) then evaluates
on the test split in a single invocation. `--stage train|evaluate` runs just one half;
`--dry-run` prints the exact s3prl commands without executing; `--extra-override` appends
verbatim `,,`-separated s3prl overrides (used for the smoke below). The default config is
`downstream/fluent_commands/config.yaml` (total_steps 200000, Adam lr 1e-4, eval every 5000
steps on dev+test); the wrapper injects only the data-root override
`config.downstream_expert.datarc.file_path=<data-root>`.

**Expected resources (1×H100).** Full run is 200 000 steps. Measured training throughput on
an otherwise-idle H100 was **~12.6 steps/s (~0.08 s/step)** for the frozen Base+ featurizer,
projecting to **~4.5–5 h** of training plus periodic dev/test evaluation. Under contention
from concurrent jobs this stretches toward the R1 estimate of **~8–12 h**; VRAM stays
**< 12 GB**. Budget PBS walltime **24:00:00** as a safe ceiling.

**Reading the output.** The evaluate stage prints `test acc: <fraction>` (from
`downstream/fluent_commands/expert.py:152`); the wrapper parses the last occurrence and
prints the canonical line
```
RESULT superb ic acc=<fraction>
```
Multiply by 100 for the leaderboard percentage. Checkpoints, config snapshots, and
per-utterance prediction/truth CSVs land in `result/downstream/<exp-name>/`; the evaluate
stage loads `dev-best.ckpt`.

**Reference (paper).** WavLM Base+ scores **IC acc = 99.00 %** in the WavLM paper
(arXiv:2110.13900, Table I). For context: Base 98.63 %, Large 99.31 %.

### Executed runs
| Date | Command (abridged) | Metric | Label |
|---|---|---|---|
| 2026-07-17 | `run_wavlm_ic.sh --exp-name smoke_ic --extra-override "config.runner.total_steps=300,,eval_step=100,,save_step=100,,log_step=50"` | `acc=0.0635` (6.35 %) | **SMOKE** |

SMOKE = a 300-step sanity run (vs. 200 000 for the full config) that proves the wrapper
trains → evaluates → prints `RESULT superb ic acc=` end-to-end. The 6.35 % accuracy is a
not-yet-trained pipeline check, **not** a benchmark number. Wall time was 92 s including
evaluation, on an H100 shared with sibling smoke jobs.

## SF — End-to-end Slot Filling

**What it measures.** SF probes how much *semantic* content a frozen SSL
representation exposes. The learnable weighted-sum over frozen WavLM hidden
states feeds a 2-layer BiLSTM + CTC head (`-d ctc -c downstream/ctc/snips.yaml`,
`character-slot` text mode) that transcribes speech into a character sequence
with **inline slot markers** (IOB-style `B-<slot> … E-<slot>` tokens around each
slot value). Two numbers are reported, both from the SUPERB tables:

- **slot-type F1** (`slot_type_f1`, *higher is better*) — did the model predict
  the right set of slot *types*?
- **slot-value CER** (`slot_value_cer`, *lower is better*) — character error rate
  of the predicted slot *values* against the reference values.

Checkpoint selection during training is on **slot-type F1** on the dev split
(the first metric in the config, `metric_higher_better: True`), saved as
`dev-best.ckpt`.

**Dataset.** Audio SNIPS (the SNIPS SLU commands re-synthesised as speech across
multiple TTS speaker voices). The `snips.yaml` recipe bakes in fixed train / dev
/ test **speaker** splits (train: Ivy, Joanna, Joey, Justin, Kendra, Kimberly,
Matthew, Salli; dev: Aditi, Amy, Geraint, Nicole; test: Brian, Emma, Raveena,
Russell). Two acquisition paths:

1. **Preprocessed zip (recommended, no mp3 tooling needed).** Download the
   ready-to-use Audio SNIPS from the s3prl-provided Google Drive and unzip:
   <https://drive.google.com/file/d/1oBRZd-PaCKz5iY3eZkXs5OB_ZZ4w7bbG/view>
   (file id `1oBRZd-PaCKz5iY3eZkXs5OB_ZZ4w7bbG`; ~11 GB zip). E.g. with
   [`gdown`](https://github.com/wkentaro/gdown):
   ```bash
   pip install gdown
   gdown 1oBRZd-PaCKz5iY3eZkXs5OB_ZZ4w7bbG -O snips.zip && unzip snips.zip
   ```
   The unzipped **`SNIPS/`** directory is your `--data-root`. It contains
   `all.iob.snips.txt` (IOB-tagged transcripts used by `character-slot` mode),
   `all-trans.txt` (plain transcripts), `slots.txt` (the slot-label inventory),
   and the `train/ valid/ test/` sub-directories of `*.wav` files (speaker-named,
   e.g. `Ivy-….wav`). A `LICENSE` file is included in the archive.

2. **Regenerate from the official release (optional, needs mp3 support).** The
   original [aws-samples/aws-lex-noisy-spoken-language-understanding](https://github.com/aws-samples/aws-lex-noisy-spoken-language-understanding)
   ships audio as **mp3**, so `sox` needs the mp3 handler
   (`apt-get install libsox-fmt-mp3`, or `yum install soxr sox-plugins-freeworld`).
   Then `./preprocess/snips_prepare_data.sh $CORPORA_DIR` (in the s3prl checkout)
   converts to wav and lays out the same directory structure. The preprocessed
   zip in path (1) exists precisely so most users can skip this.

**Run it (one command):**

```bash
bash scripts/superb/run_wavlm_sf.sh --data-root /path/to/SNIPS \
     --exp-name wavlm_base_plus_sf_full
```

`--stage all` (default) trains (`-d ctc -c downstream/ctc/snips.yaml`, frozen
upstream + learnable weighted-sum, Adam lr 1e-4, 200 000 steps, batch 32) then
evaluates `dev-best.ckpt` on the test split and prints both metrics. The `ctc`
recipe is shared by PR / SF / OOD-ASR and is specialised to SF purely by
`snips.yaml`. The wrapper sets both required overrides for you:
`config.downstream_expert.corpus.path=<data-root>` and
`config.downstream_expert.text.slots_file=<data-root>/slots.txt` (override the
latter with `SF_SLOTS_FILE=` if your slots file lives elsewhere). Other flags:
`--upstream` (default `wavlm_base_plus`), `--gpu`, `--stage all|train|evaluate`,
`--extra-override` (verbatim s3prl `-o`, `,,`-separated), `--dry-run`.

**How to read the output.** The evaluate stage prints, from
`downstream/ctc/expert.py`:

```
test slot_type_f1: <fraction>
test slot_value_cer: <fraction>
```

(plus `slot_value_wer`, `slot_edit_f1_full/part`, `wer`, `cer` — not the reported
pair). The wrapper greps the two headline metrics and appends:

```
RESULT superb sf slot_type_f1=<fraction>     # e.g. 0.9058  == 90.58 %
RESULT superb sf slot_value_cer=<fraction>   # e.g. 0.2120  == 21.20 %
```

**Both values are fractions — multiply by 100** for the percentages cited in the
papers. Artifacts land in `external/s3prl/s3prl/result/downstream/<exp-name>/`
(`dev-best.ckpt`, `config_*.yaml`, `{dev,test}-hyp.ark` / `-ref.ark`); wrapper
logs go to `logs/superb/sf/<exp-name>_{train,eval}.log`.

**Expected resources (1×H100).** `snips.yaml` defaults: `total_steps=200000`,
`batch_size=32`, LSTM+CTC head, frozen Base+ (~20 GB VRAM, R1 estimate — not
separately profiled here). **Smoke-measured throughput:** ~10 optimiser steps/s
during training and ~55 it/s during evaluation on a lightly-loaded H100 (the
SNIPS buckets are sorted longest-first, so the measured steps sit near the *slow*
end). At ~10 steps/s the full 200 000-step run is on the order of **~6–10 h**
pure training plus the periodic full-dev evaluations (every 2 000 steps) — i.e.
**materially faster than R1's initial heuristic of 1.5–3 days**. Recommended PBS
walltime **24:00:00** (generous headroom for sequence-length variation, eval
overhead, and shared-GPU slowdown).

**Reference (WavLM paper, Table I — arXiv:2110.13900).** WavLM Base+ SF
**slot_type_f1 90.58 / slot_value_cer 21.20 %**. (Base 89.38 / 22.86; Large
92.21 / 18.36.) The public SUPERB leaderboard may differ by small amounts.

**Verification status.** **SMOKE-verified end-to-end** on this hardware
(2026-07-17). A reduced 3 000-step run of the *exact* wrapper (`--stage all`)
completed train → auto-saved `dev-best.ckpt` → evaluate → both `RESULT` lines
with no manual intervention, and the model was visibly learning (dev
`slot_type_f1` climbed 0.21 → 0.72 across five dev evals). The data path
(Google-Drive zip → `--data-root`), both `-o` overrides, checkpoint selection,
and the two-metric extraction are all confirmed against real data. The full
200 000-step benchmark run is left to the end user.

> **Short-run gotcha (worth knowing).** `dev-best.ckpt` is only written when the
> dev `slot_type_f1` *exceeds* the initial best score of `0` (`ctc/expert.py`
> initialises `best_score=0` for higher-is-better metrics). A very short run
> (e.g. ≤200 steps) can leave the model predicting no slots at all, so
> `slot_type_f1` stays `0.0`, no `dev-best.ckpt` is saved, and the evaluate stage
> errors with *"checkpoint not found: …/dev-best.ckpt"*. This is expected for a
> too-short run, **not** a bug — give it enough steps to start predicting slots
> (a few hundred to a couple thousand here) and `dev-best.ckpt` appears. The full
> recipe (200 000 steps) is unaffected.

### Executed runs

| Date (UTC+8) | Upstream | Command (shrink override) | Result | Label |
|---|---|---|---|---|
| 2026-07-17 | wavlm_base_plus | `run_wavlm_sf.sh --exp-name smoke_sf_e2e --extra-override "config.runner.total_steps=3000,,config.runner.eval_step=600,,config.runner.save_step=600,,config.runner.log_step=100,,config.runner.evaluate_ratio=0.3"` | test `slot_type_f1=0.6868` (**68.68 %**) / `slot_value_cer=0.5507` (**55.07 %**) | **SMOKE** |

> **SMOKE** = 3 000 of 200 000 steps (1.5 %) with dev evaluated on a 30 % subset
> (`evaluate_ratio=0.3`) — a full end-to-end pipeline check, **not** a benchmark
> number. It proves the wrapper runs train → auto `dev-best.ckpt` → evaluate →
> both `RESULT superb sf …` lines. Wall time **7m25s** on a lightly-loaded H100.
> The metrics are already moving the right way (F1 68.68 % vs the 90.58 %
> reference, CER 55.07 % vs 21.20 %) but need the full 200 000-step run to be
> comparable. `evaluate_ratio` is a smoke-only knob; the full-run command above
> sets none of the shrink overrides.

## ER — Emotion Recognition (IEMOCAP, 5-fold)

**What it measures.** Whether the frozen WavLM Base+ representation carries enough
paralinguistic information to recognise the *emotion* of an utterance. Each IEMOCAP
utterance is mapped to one of **4 classes** — neutral, happy, angry, sad (the *excited*
class is merged into *happy*, `downstream/emotion/IEMOCAP_preprocess.py`) — and the
downstream head (a learnable weighted-sum over WavLM's 13 hidden states → mean-pooled
utterance vector → linear classifier) predicts the class. The metric is **accuracy**,
computed in `downstream/emotion/expert.py`. IEMOCAP has 5 recording sessions, so the
SUPERB protocol is **5-fold, leave-one-session-out cross-validation**: five independent
trainings, each holding out one session for test, and the reported number is the
**mean test accuracy over the 5 folds**.

**Dataset — IEMOCAP (Interactive Emotional Dyadic Motion Capture), ~12 GB.** Five
sessions of scripted + improvised dyadic conversation; the wrapper's `--data-root` must
point at the `IEMOCAP_full_release` directory (the one containing `Session1/ … Session5/`).
*Access is license-gated:* IEMOCAP is distributed by USC SAIL only after signing their
academic release form — request it at **https://sail.usc.edu/iemocap/iemocap_release.htm**
(there is no public/HF mirror). No preprocessing or download step is needed once the
release is unpacked: the **fold metadata ships inside s3prl** as JSON
(`downstream/emotion/meta_data/Session{1..5}/{train,test}_meta_data.json`), and each JSON's
`path` fields are resolved relative to the data root, so pointing `--data-root` at the
unpacked release is sufficient.

**Run (one command).**
```bash
bash scripts/superb/run_wavlm_er.sh \
  --data-root /path/to/IEMOCAP_full_release \
  --fold all \
  --exp-name wavlm_base_plus_er_full
```
Runs all five folds end-to-end: for each fold it trains (frozen upstream, learnable
weighted-sum — no `-f`/`-l`, per SUPERB) then evaluates that fold's test session, and
finally prints the per-fold and mean accuracy. `--fold fold1|…|fold5` runs a single fold;
`--stage train|evaluate` runs just one half; `--dry-run` prints the exact s3prl commands
without executing; `--extra-override` appends verbatim `,,`-separated s3prl overrides (used
for the smoke below). The default config is `downstream/emotion/config.yaml` (total_steps
30000 per fold, Adam lr 1e-4, grad-accum 8 × batch 4, eval every 1000 steps on dev+test).
The wrapper injects two overrides per fold: the data root
`config.downstream_expert.datarc.root=<data-root>` and the held-out fold
`config.downstream_expert.datarc.test_fold='foldN'` (the single quotes are required so
s3prl's `override()` evaluates the value to the clean string `foldN`, `utility/helper.py`).

**Expected resources (1×H100).** Full run is 5 folds × 30 000 optimiser steps (each step =
8 grad-accum micro-batches of 4). Smoke throughput was **~1.7 s/step in steady state**
(best window ~1.5 s/step; ~3.15 s/step averaged over the 200-step smoke, which includes the
two in-training dev+test evals and a transient contention spike), measured under **heavy
4-way H100 contention** from sibling smoke jobs. On a **dedicated** node the R1 estimate of
**~3–5 h/fold → ~15–25 h total** applies; at the contended smoke rate it would be roughly
3× that. VRAM stays **< 10 GB** (frozen upstream, small utterance-level head). Budget PBS
walltime **36:00:00** for the 5-fold sweep on a dedicated node (if the node is shared,
submit folds as separate jobs or raise the ceiling).

**Reading the output.** The evaluate stage of each fold prints `test acc: <fraction>` (from
`downstream/emotion/expert.py:137`, evaluate defaults to the `test` split). The wrapper
parses the last occurrence per fold and prints the canonical lines
```
RESULT superb er acc_fold1=<fraction>   # one per executed fold
RESULT superb er acc_mean=<fraction>    # mean over the executed folds
```
Multiply by 100 for the leaderboard percentage. Per-fold checkpoints, config snapshots, and
per-utterance prediction/truth files land in `result/downstream/<exp-name>_foldN/`; the
evaluate stage loads that fold's `dev-best.ckpt`.

**Reference (paper).** WavLM Base+ scores **ER acc = 68.65 %** in the WavLM paper
(arXiv:2110.13900, Table I; 5-fold IEMOCAP mean). For context: Base 65.94 %, Large 70.62 %.

### Executed runs
| Date | Command (abridged) | Metric | Label |
|---|---|---|---|
| 2026-07-17 | `run_wavlm_er.sh --fold fold1 --exp-name smoke_er --extra-override "config.runner.total_steps=200,,eval_step=100,,save_step=100,,log_step=50"` | `acc_fold1=0.4470` (44.70 %) | **SMOKE** |

SMOKE = a single-fold, 200-step sanity run (vs. 30 000 steps/fold × 5 folds for the full
config) that proves the wrapper trains → evaluates → prints `RESULT superb er acc_fold1=`
end-to-end. The 44.70 % accuracy is a barely-trained pipeline check, **not** a benchmark
number (chance on 4 balanced classes ≈ 25 %; it already clears chance after 200 steps,
confirming the head learns). Wall time was **~14 min 33 s** (start 06:10:48 → end 06:25:21)
including WavLM download reuse, data preload, 200 training steps and the separate evaluate
pass, on an H100 shared with four sibling smoke jobs.
