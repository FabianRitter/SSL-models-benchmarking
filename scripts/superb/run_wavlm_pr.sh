#!/usr/bin/env bash
# =============================================================================
# SUPERB PR — Phoneme Recognition · upstream: WavLM (s3prl)
# Trains a frozen-upstream + weighted-sum CTC phoneme recognizer on LibriSpeech
# and evaluates it end-to-end with one command.
# Generic local/conda execution — no scheduler assumptions.
#
# Usage:
#   bash scripts/superb/run_wavlm_pr.sh --data-root PATH [options]
# Options:
#   --data-root PATH   LibriSpeech root with train-clean-100/ dev-clean/ test-clean/
#                      (REQUIRED; see docs/superb.md#pr)
#   --upstream NAME    s3prl upstream (default: wavlm_base_plus)
#   --gpu IDS          CUDA_VISIBLE_DEVICES value (default: 0)
#   --exp-name NAME    experiment name (default: <upstream>_pr)
#   --stage STAGE      all | train | evaluate (default: all)
#   --extra-override S extra s3prl -o overrides, appended verbatim after the
#                      data-root override with the ',,' separator (default: none)
#   --dry-run          print commands without executing
# Env overrides:
#   S3PRL_ROOT           s3prl checkout (default: <repo>/external/s3prl)
#   SSL_BENCH_CONDA_ENV  conda env name/path (default: ssl-bench-s3prl)
#
# Downstream recipe: -d ctc  -c downstream/ctc/libriphone.yaml  (the 'ctc' recipe
#   serves PR/SF/OOD-ASR; PR is selected by this config). Data-root override key:
#   config.downstream_expert.corpus.path
# Metric: phoneme error rate 'per' (fraction; PER% = per*100). Printed by
#   ctc/expert.py as "test per: <val>"; lower is better.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UPSTREAM="wavlm_base_plus"; GPU="${CUDA_VISIBLE_DEVICES:-0}"; DATA_ROOT=""
EXP_NAME=""; STAGE="all"; EXTRA_O=""; DRY_RUN=0
PR_CONFIG="downstream/ctc/libriphone.yaml"

while [[ $# -gt 0 ]]; do case "$1" in
  --data-root) DATA_ROOT="$2"; shift 2;;  --upstream) UPSTREAM="$2"; shift 2;;
  --gpu) GPU="$2"; shift 2;;              --exp-name) EXP_NAME="$2"; shift 2;;
  --stage) STAGE="$2"; shift 2;;          --extra-override) EXTRA_O="$2"; shift 2;;
  --dry-run) DRY_RUN=1; shift;;
  *) echo "unknown option: $1" >&2; exit 2;; esac; done

[[ -n "$DATA_ROOT" ]] || { echo "ERROR: --data-root is required (see docs/superb.md#pr)" >&2; exit 2; }
case "$STAGE" in all|train|evaluate) ;; *) echo "ERROR: --stage must be all|train|evaluate" >&2; exit 2;; esac
EXP_NAME="${EXP_NAME:-${UPSTREAM}_pr}"
S3PRL_ROOT="${S3PRL_ROOT:-$REPO_ROOT/external/s3prl}"
[[ -d "$S3PRL_ROOT/s3prl" ]] || { echo "ERROR: s3prl not found at $S3PRL_ROOT (set S3PRL_ROOT or git submodule update --init)" >&2; exit 2; }

# LibriSpeech must contain the three PR splits (train-clean-100 / dev-clean / test-clean).
for split in train-clean-100 dev-clean test-clean; do
  [[ -d "$DATA_ROOT/$split" ]] || { echo "ERROR: '$split' not found under --data-root '$DATA_ROOT'" >&2; exit 2; }
done

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SSL_BENCH_CONDA_ENV:-ssl-bench-s3prl}"
export CUDA_VISIBLE_DEVICES="$GPU"

LOG_DIR="$REPO_ROOT/logs/superb/pr"; mkdir -p "$LOG_DIR"
TRAIN_LOG="$LOG_DIR/${EXP_NAME}_train.log"
EVAL_LOG="$LOG_DIR/${EXP_NAME}_eval.log"
cd "$S3PRL_ROOT/s3prl"

# Data-root override, plus any extra (smoke-shrink) overrides appended verbatim.
OVERRIDE="config.downstream_expert.corpus.path=${DATA_ROOT}"
[[ -n "$EXTRA_O" ]] && OVERRIDE="${OVERRIDE},,${EXTRA_O}"

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
    python3 run_downstream.py -m train -u "$UPSTREAM" -d ctc -c "$PR_CONFIG" -n "$EXP_NAME" -o "$OVERRIDE"
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

# --- Report the metric (only when an evaluation ran) -------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "evaluate" ]]; then
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "RESULT superb pr per=<dry-run>"
  else
    PER="$(grep -E 'test per:' "$EVAL_LOG" | tail -n1 | sed -E 's/.*test per:[[:space:]]*//' | awk '{print $1}' || true)"
    [[ -n "$PER" ]] || { echo "ERROR: could not find 'test per:' in $EVAL_LOG" >&2; exit 1; }
    PER_PCT="$(python3 -c "print(f'{float('$PER')*100:.2f}')" 2>/dev/null || echo '?')"
    echo "PR test PER = ${PER} (fraction) = ${PER_PCT}% | exp=${EXP_NAME} | logs: ${EVAL_LOG}"
    echo "RESULT superb pr per=${PER}"
  fi
fi
