#!/usr/bin/env bash
# =============================================================================
# SUPERB ASR — Automatic Speech Recognition · upstream: WavLM (s3prl)
# Prepares the LibriSpeech length-bucket table, trains the LSTM+CTC head on
# frozen WavLM features, and evaluates WER on test-clean — one command.
# Generic local/conda execution — no scheduler assumptions.
#
# Usage:
#   bash scripts/superb/run_wavlm_asr.sh --data-root PATH [options]
# Options:
#   --data-root PATH   LibriSpeech root (REQUIRED; dir name must contain
#                      "LibriSpeech"; see docs/superb.md#asr)
#   --upstream NAME    s3prl upstream (default: wavlm_base_plus)
#   --gpu IDS          CUDA_VISIBLE_DEVICES value (default: 0)
#   --exp-name NAME    experiment name (default: <upstream>_asr)
#   --stage STAGE      all | prep | train | evaluate (default: all)
#                        all      = prep-if-missing -> train -> evaluate
#                        prep     = (re)generate the length-bucket table only
#                        train    = prep-if-missing -> train
#                        evaluate = evaluate an existing dev-clean-best.ckpt
#   --extra-override S extra s3prl -o overrides, appended verbatim (default: none)
#   --dry-run          print commands without executing
# Env overrides:
#   S3PRL_ROOT           s3prl checkout (default: <repo>/external/s3prl)
#   SSL_BENCH_CONDA_ENV  conda env name/path (default: ssl-bench-s3prl)
#   PREP_NJOBS           parallel jobs for the bucket-length prep (default: 12)
#
# Notes:
#   * No language-model decoding (matches the WavLM paper's no-LM ASR row).
#   * Eval checkpoint is dev-clean-best.ckpt (the runner selects on dev-clean),
#     scored on the test-clean split.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

UPSTREAM="wavlm_base_plus"; GPU="${CUDA_VISIBLE_DEVICES:-0}"; DATA_ROOT=""
EXP_NAME=""; STAGE="all"; EXTRA_O=""; DRY_RUN=0
while [[ $# -gt 0 ]]; do case "$1" in
  --data-root) DATA_ROOT="$2"; shift 2;;  --upstream) UPSTREAM="$2"; shift 2;;
  --gpu) GPU="$2"; shift 2;;              --exp-name) EXP_NAME="$2"; shift 2;;
  --stage) STAGE="$2"; shift 2;;          --extra-override) EXTRA_O="$2"; shift 2;;
  --dry-run) DRY_RUN=1; shift;;
  *) echo "unknown option: $1" >&2; exit 2;; esac; done

case "$STAGE" in all|prep|train|evaluate) ;; *)
  echo "ERROR: --stage must be one of: all | prep | train | evaluate" >&2; exit 2;; esac
[[ -n "$DATA_ROOT" ]] || { echo "ERROR: --data-root is required (see docs/superb.md#asr)" >&2; exit 2; }
EXP_NAME="${EXP_NAME:-${UPSTREAM}_asr}"
PREP_NJOBS="${PREP_NJOBS:-12}"

S3PRL_ROOT="${S3PRL_ROOT:-$REPO_ROOT/external/s3prl}"
[[ -d "$S3PRL_ROOT/s3prl" ]] || { echo "ERROR: s3prl not found at $S3PRL_ROOT (set S3PRL_ROOT or git submodule update --init)" >&2; exit 2; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SSL_BENCH_CONDA_ENV:-ssl-bench-s3prl}"
export CUDA_VISIBLE_DEVICES="$GPU"

LOG_DIR="$REPO_ROOT/logs/superb/asr"; mkdir -p "$LOG_DIR"
PREP_LOG="$LOG_DIR/${EXP_NAME}_prep.log"
TRAIN_LOG="$LOG_DIR/${EXP_NAME}_train.log"
EVAL_LOG="$LOG_DIR/${EXP_NAME}_eval.log"
cd "$S3PRL_ROOT/s3prl"

# Length-bucket table (config default: ./data/librispeech/len_for_bucket).
BUCKET_DIR="data/librispeech/len_for_bucket"
CKPT="result/downstream/$EXP_NAME/dev-clean-best.ckpt"

# Base data-root override; append user extras with the ',,' field separator.
OVERRIDE="config.downstream_expert.datarc.libri_root=$DATA_ROOT"
[[ -n "$EXTRA_O" ]] && OVERRIDE="${OVERRIDE},,${EXTRA_O}"

echo "=============================================================================="
echo " SUPERB ASR | upstream=$UPSTREAM exp=$EXP_NAME stage=$STAGE gpu=$GPU"
echo " s3prl=$S3PRL_ROOT/s3prl  data-root=$DATA_ROOT"
echo " override: $OVERRIDE"
echo "=============================================================================="

run() { echo "+ $*"; [[ $DRY_RUN -eq 1 ]] && return 0; "$@"; }

bucket_ready() {
  [[ -f "$BUCKET_DIR/train-clean-100.csv" \
  && -f "$BUCKET_DIR/dev-clean.csv" \
  && -f "$BUCKET_DIR/test-clean.csv" ]]
}

do_prep() {
  # generate_len_for_bucket.py is interactive: it lists the LibriSpeech splits
  # and reads their indices from stdin. For SUPERB ASR we need
  #   0 = train-clean-100, 3 = dev-clean, 5 = test-clean.
  # (The script requires "librispeech" to appear in --data-root's path.)
  [[ $DRY_RUN -eq 1 || -d "$DATA_ROOT" ]] || { echo "ERROR: --data-root does not exist: $DATA_ROOT" >&2; exit 3; }
  echo "+ echo '0 3 5' | python3 preprocess/generate_len_for_bucket.py -i $DATA_ROOT -o data/librispeech -a .flac --n_jobs $PREP_NJOBS"
  [[ $DRY_RUN -eq 1 ]] && return 0
  echo "0 3 5" | python3 preprocess/generate_len_for_bucket.py \
      -i "$DATA_ROOT" -o data/librispeech -a .flac --n_jobs "$PREP_NJOBS" 2>&1 | tee "$PREP_LOG"
  bucket_ready || { echo "ERROR: prep did not produce the expected CSVs in $BUCKET_DIR" >&2; exit 3; }
}

prep_if_missing() {
  if bucket_ready; then
    echo "+ bucket table present ($BUCKET_DIR) — skipping prep"
  else
    do_prep
  fi
}

do_train() {
  [[ $DRY_RUN -eq 1 || -d "$DATA_ROOT" ]] || { echo "ERROR: --data-root does not exist: $DATA_ROOT" >&2; exit 3; }
  run python3 run_downstream.py -m train -u "$UPSTREAM" -d asr -n "$EXP_NAME" \
      -o "$OVERRIDE" 2>&1 | tee "$TRAIN_LOG"
}

do_evaluate() {
  # No LM. Config (incl. libri_root/bucket_file) is restored from the ckpt;
  # -m evaluate / -t test-clean are preserved over the ckpt's stored Args.
  [[ $DRY_RUN -eq 1 || -f "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT (run training first)" >&2; exit 3; }
  run python3 run_downstream.py -m evaluate -t "test-clean" -e "$CKPT" 2>&1 | tee "$EVAL_LOG"
  [[ $DRY_RUN -eq 1 ]] && { echo "RESULT superb asr wer=<dry-run>"; return 0; }
  # downstream/asr/expert.py prints WER already as a PERCENTAGE (0-100),
  # e.g. "test-clean wer: 5.59"  (== 5.59 %). Report it verbatim, no scaling.
  local wer
  wer="$(grep -oE "test-clean wer: [0-9.eE+-]+" "$EVAL_LOG" | tail -1 | sed -E 's/.*wer: //')"
  [[ -n "$wer" ]] || { echo "ERROR: could not parse 'test-clean wer:' from $EVAL_LOG" >&2; exit 4; }
  echo "test-clean WER = ${wer}% (no LM)" | tee -a "$EVAL_LOG"
  echo "RESULT superb asr wer=$wer" | tee -a "$EVAL_LOG"
}

case "$STAGE" in
  all)      prep_if_missing; do_train; do_evaluate;;
  prep)     do_prep;;
  train)    prep_if_missing; do_train;;
  evaluate) do_evaluate;;
esac
