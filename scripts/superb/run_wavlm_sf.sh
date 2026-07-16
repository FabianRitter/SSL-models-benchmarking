#!/usr/bin/env bash
# =============================================================================
# SUPERB SF — End-to-end Slot Filling · upstream: WavLM (s3prl)
# Trains a frozen-upstream + weighted-sum LSTM/CTC slot-filling model on Audio
# SNIPS and evaluates it end-to-end with one command.
# Generic local/conda execution — no scheduler assumptions.
#
# Usage:
#   bash scripts/superb/run_wavlm_sf.sh --data-root PATH [options]
# Options:
#   --data-root PATH   Audio SNIPS root, containing all.iob.snips.txt, slots.txt
#                      and the train/ valid/ test/ wav sub-dirs
#                      (REQUIRED; see docs/superb.md#sf)
#   --upstream NAME    s3prl upstream (default: wavlm_base_plus)
#   --gpu IDS          CUDA_VISIBLE_DEVICES value (default: 0)
#   --exp-name NAME    experiment name (default: <upstream>_sf)
#   --stage STAGE      all | train | evaluate (default: all)
#   --extra-override S extra s3prl -o overrides, appended verbatim after the
#                      data-root overrides with the ',,' separator (default: none)
#   --dry-run          print commands without executing
# Env overrides:
#   S3PRL_ROOT           s3prl checkout (default: <repo>/external/s3prl)
#   SSL_BENCH_CONDA_ENV  conda env name/path (default: ssl-bench-s3prl)
#   SF_SLOTS_FILE        slots file path (default: <data-root>/slots.txt)
#
# Downstream recipe: -d ctc  -c downstream/ctc/snips.yaml  (the 'ctc' recipe
#   serves PR/SF/OOD-ASR; SF is selected by this config, character-slot text
#   mode). Data-root override keys:
#     config.downstream_expert.corpus.path      -> SNIPS root
#     config.downstream_expert.text.slots_file  -> <SNIPS>/slots.txt
# Metrics (both SUPERB-reported): slot_type_f1 (F1, fraction; higher better) and
#   slot_value_cer (CER, fraction; lower better). Printed by ctc/expert.py as
#   "test slot_type_f1: <val>" / "test slot_value_cer: <val>". Multiply by 100
#   to compare against the WavLM paper (90.58 / 21.20 for Base+).
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UPSTREAM="wavlm_base_plus"; GPU="${CUDA_VISIBLE_DEVICES:-0}"; DATA_ROOT=""
EXP_NAME=""; STAGE="all"; EXTRA_O=""; DRY_RUN=0
SF_CONFIG="downstream/ctc/snips.yaml"

while [[ $# -gt 0 ]]; do case "$1" in
  --data-root) DATA_ROOT="$2"; shift 2;;  --upstream) UPSTREAM="$2"; shift 2;;
  --gpu) GPU="$2"; shift 2;;              --exp-name) EXP_NAME="$2"; shift 2;;
  --stage) STAGE="$2"; shift 2;;          --extra-override) EXTRA_O="$2"; shift 2;;
  --dry-run) DRY_RUN=1; shift;;
  *) echo "unknown option: $1" >&2; exit 2;; esac; done

[[ -n "$DATA_ROOT" ]] || { echo "ERROR: --data-root is required (see docs/superb.md#sf)" >&2; exit 2; }
case "$STAGE" in all|train|evaluate) ;; *) echo "ERROR: --stage must be all|train|evaluate" >&2; exit 2;; esac
EXP_NAME="${EXP_NAME:-${UPSTREAM}_sf}"
DATA_ROOT="${DATA_ROOT%/}"                         # strip a trailing slash (keeps -o clean)
SLOTS_FILE="${SF_SLOTS_FILE:-$DATA_ROOT/slots.txt}"
S3PRL_ROOT="${S3PRL_ROOT:-$REPO_ROOT/external/s3prl}"
[[ -d "$S3PRL_ROOT/s3prl" ]] || { echo "ERROR: s3prl not found at $S3PRL_ROOT (set S3PRL_ROOT or git submodule update --init)" >&2; exit 2; }

# The character-slot text mode reads all.iob.snips.txt + slots.txt from the root,
# and the corpus loader globs train/ valid/ test/ for *.wav. Fail early & clearly.
if [[ $DRY_RUN -eq 0 ]]; then
  [[ -f "$SLOTS_FILE" ]] || { echo "ERROR: slots file not found: $SLOTS_FILE (point --data-root at the SNIPS root, or set SF_SLOTS_FILE)" >&2; exit 2; }
  [[ -f "$DATA_ROOT/all.iob.snips.txt" ]] || { echo "ERROR: 'all.iob.snips.txt' not found under --data-root '$DATA_ROOT'" >&2; exit 2; }
  for split in train valid test; do
    [[ -d "$DATA_ROOT/$split" ]] || { echo "ERROR: split dir '$split' not found under --data-root '$DATA_ROOT'" >&2; exit 2; }
  done
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SSL_BENCH_CONDA_ENV:-ssl-bench-s3prl}"
export CUDA_VISIBLE_DEVICES="$GPU"

LOG_DIR="$REPO_ROOT/logs/superb/sf"; mkdir -p "$LOG_DIR"
TRAIN_LOG="$LOG_DIR/${EXP_NAME}_train.log"
EVAL_LOG="$LOG_DIR/${EXP_NAME}_eval.log"
cd "$S3PRL_ROOT/s3prl"

# Two data-root overrides (corpus path + slots file), plus any extra (smoke-shrink)
# overrides appended verbatim. Paths only — no quoting needed, which keeps the
# string safe across the run() argv boundary (no bash -c / eval layers).
OVERRIDE="config.downstream_expert.corpus.path=${DATA_ROOT},,config.downstream_expert.text.slots_file=${SLOTS_FILE}"
[[ -n "$EXTRA_O" ]] && OVERRIDE="${OVERRIDE},,${EXTRA_O}"

echo "=============================================================================="
echo " SUPERB SF | upstream=$UPSTREAM exp=$EXP_NAME stage=$STAGE gpu=$GPU"
echo " s3prl=$S3PRL_ROOT/s3prl  data-root=$DATA_ROOT"
echo " override: $OVERRIDE"
echo "=============================================================================="

# run_logged LOGFILE cmd...  — echoes the command, tees output; pipefail keeps
# the exit code honest (the python call's status, not tee's).
run_logged() {
  local log="$1"; shift
  echo "+ $* 2>&1 | tee $log"
  [[ $DRY_RUN -eq 1 ]] && return 0
  "$@" 2>&1 | tee "$log"
}

do_train() {
  run_logged "$TRAIN_LOG" \
    python3 run_downstream.py -m train -u "$UPSTREAM" -d ctc -c "$SF_CONFIG" -n "$EXP_NAME" -o "$OVERRIDE"
}

do_evaluate() {
  local ckpt="result/downstream/${EXP_NAME}/dev-best.ckpt"
  if [[ $DRY_RUN -eq 0 && ! -f "$ckpt" ]]; then
    echo "ERROR: checkpoint not found: $S3PRL_ROOT/s3prl/$ckpt (did training save dev-best.ckpt?)" >&2
    exit 1
  fi
  run_logged "$EVAL_LOG" \
    python3 run_downstream.py -m evaluate -t test -e "$ckpt"
}

case "$STAGE" in
  train)    do_train ;;
  evaluate) do_evaluate ;;
  all)      do_train; do_evaluate ;;
esac

# --- Report BOTH metrics (only when an evaluation ran) -----------------------
if [[ "$STAGE" == "all" || "$STAGE" == "evaluate" ]]; then
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "RESULT superb sf slot_type_f1=<dry-run>"
    echo "RESULT superb sf slot_value_cer=<dry-run>"
  else
    # expert.py prints e.g. "test slot_type_f1: 0.9058" (fraction) and
    # "test slot_value_cer: 0.2120" (fraction). Extract each verbatim.
    F1="$(grep -E 'test slot_type_f1:' "$EVAL_LOG" | tail -n1 | sed -E 's/.*test slot_type_f1:[[:space:]]*//' | awk '{print $1}' || true)"
    CER="$(grep -E 'test slot_value_cer:' "$EVAL_LOG" | tail -n1 | sed -E 's/.*test slot_value_cer:[[:space:]]*//' | awk '{print $1}' || true)"
    [[ -n "$F1" ]]  || { echo "ERROR: could not find 'test slot_type_f1:' in $EVAL_LOG" >&2; exit 1; }
    [[ -n "$CER" ]] || { echo "ERROR: could not find 'test slot_value_cer:' in $EVAL_LOG" >&2; exit 1; }
    F1_PCT="$(awk -v v="$F1" 'BEGIN{printf "%.2f", v*100}' 2>/dev/null || echo '?')"
    CER_PCT="$(awk -v v="$CER" 'BEGIN{printf "%.2f", v*100}' 2>/dev/null || echo '?')"
    echo "SF test slot_type_f1  = ${F1} (fraction) = ${F1_PCT}% (higher better) | exp=${EXP_NAME}"
    echo "SF test slot_value_cer = ${CER} (fraction) = ${CER_PCT}% (lower better)  | logs: ${EVAL_LOG}"
    echo "RESULT superb sf slot_type_f1=${F1}"
    echo "RESULT superb sf slot_value_cer=${CER}"
  fi
fi
