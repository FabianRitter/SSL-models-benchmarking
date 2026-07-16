#!/usr/bin/env bash
# =============================================================================
# SUPERB IC — Intent Classification · upstream: WavLM (s3prl)
# Trains an utterance-level classifier on frozen WavLM features (weighted-sum
# over hidden states) and evaluates end-to-end with one command.
# Generic local/conda execution — no scheduler assumptions.
#
# Usage:
#   bash scripts/superb/run_wavlm_ic.sh --data-root PATH [options]
# Options:
#   --data-root PATH   Fluent Speech Commands dataset root (REQUIRED; see docs/superb.md#ic).
#                      Must DIRECTLY contain wavs/ and data/{train,valid,test}_data.csv
#   --upstream NAME    s3prl upstream (default: wavlm_base_plus)
#   --gpu IDS          CUDA_VISIBLE_DEVICES value (default: 0)
#   --exp-name NAME    experiment name (default: <upstream>_ic)
#   --stage STAGE      all | train | evaluate (default: all)
#   --extra-override S extra s3prl -o overrides, appended verbatim after ,, (default: none)
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
[[ -n "$DATA_ROOT" ]] || { echo "ERROR: --data-root is required (see docs/superb.md#ic)" >&2; exit 2; }
EXP_NAME="${EXP_NAME:-${UPSTREAM}_ic}"
S3PRL_ROOT="${S3PRL_ROOT:-$REPO_ROOT/external/s3prl}"
[[ -d "$S3PRL_ROOT/s3prl" ]] || { echo "ERROR: s3prl not found at $S3PRL_ROOT (set S3PRL_ROOT or git submodule update --init)" >&2; exit 2; }
# Data-root sanity: the classic mistake is a double-nested dir. Require the CSVs
# to sit DIRECTLY under <data-root>/data/ (this is the path fluent_commands reads).
[[ -f "$DATA_ROOT/data/train_data.csv" ]] || { echo "ERROR: '$DATA_ROOT/data/train_data.csv' not found — --data-root must DIRECTLY contain wavs/ and data/{train,valid,test}_data.csv (see docs/superb.md#ic)" >&2; exit 2; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SSL_BENCH_CONDA_ENV:-ssl-bench-s3prl}"
export CUDA_VISIBLE_DEVICES="$GPU"
LOG_DIR="$REPO_ROOT/logs/superb/ic"; mkdir -p "$LOG_DIR"
cd "$S3PRL_ROOT/s3prl"
run() { echo "+ $*"; [[ $DRY_RUN -eq 1 ]] || "$@"; }

# Data-root override key verified in downstream/fluent_commands/config.yaml
# (config.downstream_expert.datarc.file_path). Extra overrides appended verbatim.
OVERRIDE="config.downstream_expert.datarc.file_path=${DATA_ROOT}"
[[ -n "$EXTRA_O" ]] && OVERRIDE="${OVERRIDE},,${EXTRA_O}"

TRAIN_LOG="$LOG_DIR/${EXP_NAME}_train.log"
EVAL_LOG="$LOG_DIR/${EXP_NAME}_eval.log"
CKPT="result/downstream/${EXP_NAME}/dev-best.ckpt"

# ---- Train stage (frozen upstream, learnable weighted-sum; no -f/-l per SUPERB) ----
if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
  run bash -c "set -o pipefail; python3 run_downstream.py -m train -u '${UPSTREAM}' -d fluent_commands -n '${EXP_NAME}' -o '${OVERRIDE}' 2>&1 | tee '${TRAIN_LOG}'"
fi

# ---- Evaluate stage (default split=test → prints 'test acc: <val>') ----
if [[ "$STAGE" == "all" || "$STAGE" == "evaluate" ]]; then
  [[ $DRY_RUN -eq 1 || -f "$CKPT" ]] || { echo "ERROR: checkpoint not found: $(pwd)/$CKPT (run the train stage first)" >&2; exit 2; }
  run bash -c "set -o pipefail; python3 run_downstream.py -m evaluate -e '${CKPT}' 2>&1 | tee '${EVAL_LOG}'"
  if [[ $DRY_RUN -eq 0 ]]; then
    # 'test acc: <val>' printed by downstream/fluent_commands/expert.py:152 (acc = exact
    # match over the action/object/location slots). Take the last occurrence.
    ACC="$(grep -oE 'test acc: [0-9.]+' "$EVAL_LOG" | tail -1 | awk '{print $NF}')"
    [[ -n "$ACC" ]] || { echo "ERROR: could not parse 'test acc:' from $EVAL_LOG" >&2; exit 1; }
    echo "RESULT superb ic acc=${ACC}"
  fi
fi
