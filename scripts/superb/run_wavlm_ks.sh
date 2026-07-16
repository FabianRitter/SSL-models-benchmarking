#!/usr/bin/env bash
# =============================================================================
# SUPERB KS — Keyword Spotting · upstream: WavLM (s3prl)
# Trains a downstream classifier on a frozen upstream and evaluates it
# end-to-end with one command.
# Generic local/conda execution — no scheduler assumptions.
#
# Dataset: Google Speech Commands v0.01 (see docs/superb.md#ks-keyword-spotting).
# The --data-root is the PARENT directory that contains BOTH extracted tarballs:
#   <data-root>/speech_commands_v0.01/           (train+dev tarball)
#   <data-root>/speech_commands_test_set_v0.01/  (official test-set tarball)
#
# Usage:
#   bash scripts/superb/run_wavlm_ks.sh --data-root PATH [options]
# Options:
#   --data-root PATH   Speech Commands parent dir (REQUIRED; see docs/superb.md#ks-keyword-spotting)
#   --upstream NAME    s3prl upstream (default: wavlm_base_plus)
#   --gpu IDS          CUDA_VISIBLE_DEVICES value (default: 0)
#   --exp-name NAME    experiment name (default: <upstream>_ks)
#   --stage STAGE      all | train | evaluate (default: all)
#   --extra-override S extra s3prl -o overrides, appended verbatim (default: none)
#   --dry-run          print commands without executing
# Env overrides:
#   S3PRL_ROOT           s3prl checkout (default: <repo>/external/s3prl)
#   SSL_BENCH_CONDA_ENV  conda env name/path (default: ssl-bench-s3prl)
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

[[ -n "$DATA_ROOT" ]] || { echo "ERROR: --data-root is required (see docs/superb.md#ks-keyword-spotting)" >&2; exit 2; }
case "$STAGE" in all|train|evaluate) ;; *) echo "ERROR: --stage must be all|train|evaluate (got '$STAGE')" >&2; exit 2;; esac

# KS uses two roots: the train/dev tarball and the official test-set tarball.
TRAIN_ROOT="$DATA_ROOT/speech_commands_v0.01"
TEST_ROOT="$DATA_ROOT/speech_commands_test_set_v0.01"
if [[ $DRY_RUN -eq 0 ]]; then
  [[ -d "$TRAIN_ROOT" ]] || { echo "ERROR: missing '$TRAIN_ROOT'. --data-root must contain speech_commands_v0.01/ and speech_commands_test_set_v0.01/ (see docs/superb.md#ks-keyword-spotting)" >&2; exit 2; }
  [[ -d "$TEST_ROOT"  ]] || { echo "ERROR: missing '$TEST_ROOT'. --data-root must contain speech_commands_v0.01/ and speech_commands_test_set_v0.01/ (see docs/superb.md#ks-keyword-spotting)" >&2; exit 2; }
fi

EXP_NAME="${EXP_NAME:-${UPSTREAM}_ks}"
S3PRL_ROOT="${S3PRL_ROOT:-$REPO_ROOT/external/s3prl}"
[[ -d "$S3PRL_ROOT/s3prl" ]] || { echo "ERROR: s3prl not found at $S3PRL_ROOT (set S3PRL_ROOT or git submodule update --init)" >&2; exit 2; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SSL_BENCH_CONDA_ENV:-ssl-bench-s3prl}"
export CUDA_VISIBLE_DEVICES="$GPU"

LOG_DIR="$REPO_ROOT/logs/superb/ks"; mkdir -p "$LOG_DIR"
TRAIN_LOG="$LOG_DIR/${EXP_NAME}_train.log"
EVAL_LOG="$LOG_DIR/${EXP_NAME}_eval.log"
CKPT="result/downstream/$EXP_NAME/dev-best.ckpt"
cd "$S3PRL_ROOT/s3prl"

# Data-root override (two roots). Eval restores this from the ckpt Config, so it
# is only needed at train time.
OVERRIDE="config.downstream_expert.datarc.speech_commands_root=$TRAIN_ROOT,,config.downstream_expert.datarc.speech_commands_test_root=$TEST_ROOT"
[[ -n "$EXTRA_O" ]] && OVERRIDE="$OVERRIDE,,$EXTRA_O"

run() { echo "+ $*"; [[ $DRY_RUN -eq 1 ]] || "$@"; }
# Run a command, teeing combined stdout+stderr to a log; pipefail keeps the
# python exit code (not tee's) so set -e still aborts on a real failure.
run_logged() {
  local logf="$1"; shift
  echo "+ $* 2>&1 | tee $logf"
  [[ $DRY_RUN -eq 1 ]] && return 0
  "$@" 2>&1 | tee "$logf"
}

echo "==> SUPERB KS | upstream=$UPSTREAM exp=$EXP_NAME stage=$STAGE gpu=$GPU"
echo "==> s3prl=$S3PRL_ROOT/s3prl  data-root=$DATA_ROOT"

if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
  echo "==> [train] result/downstream/$EXP_NAME"
  run_logged "$TRAIN_LOG" \
    python3 run_downstream.py -m train -u "$UPSTREAM" -d speech_commands -n "$EXP_NAME" -o "$OVERRIDE"
fi

if [[ "$STAGE" == "all" || "$STAGE" == "evaluate" ]]; then
  if [[ $DRY_RUN -eq 0 && ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint not found: $S3PRL_ROOT/s3prl/$CKPT (run train stage first)" >&2; exit 2
  fi
  echo "==> [evaluate] $CKPT (split=test)"
  run_logged "$EVAL_LOG" \
    python3 run_downstream.py -m evaluate -e "$CKPT"

  if [[ $DRY_RUN -eq 0 ]]; then
    # expert.py prints "test acc: <fraction>" (speech_commands/expert.py:131)
    ACC="$(grep -oE 'test acc: [0-9.]+' "$EVAL_LOG" | tail -1 | awk '{print $NF}')"
    [[ -n "$ACC" ]] || { echo "ERROR: could not parse 'test acc:' from $EVAL_LOG" >&2; exit 3; }
    PCT="$(awk -v a="$ACC" 'BEGIN{printf "%.2f", a*100}')"
    echo "==> KS test accuracy: $ACC (${PCT}%)"
    echo "RESULT superb ks acc=$ACC"
  fi
fi
