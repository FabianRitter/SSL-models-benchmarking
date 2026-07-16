#!/usr/bin/env bash
# =============================================================================
# SUPERB ASV — Automatic Speaker Verification (VoxCeleb1) · upstream: WavLM (s3prl)
# Trains an X-Vector + AM-Softmax speaker-verification head on frozen WavLM
# features (learnable weighted-sum), then evaluates EVERY saved checkpoint on the
# VoxCeleb1 test trials and reports the best EER. There is no dev-best checkpoint
# under the VoxCeleb1 setting (eval_dataloaders: []); the recipe saves
# states-<step>.ckpt and scores them with the shipped test_expdir.sh loop.
# Generic local/conda execution — no scheduler assumptions.
#
# Usage:
#   bash scripts/superb/run_wavlm_asv.sh --data-root PATH [options]
# Options:
#   --data-root PATH   VoxCeleb1 root (REQUIRED; same audio as SID). Contains the
#                      wav tree the trial list voxceleb1_test_v2.txt refers to,
#                      i.e. <root>/wav/id1XXXX/<video>/<utt>.wav. See docs/superb.md#asv.
#   --upstream NAME    s3prl upstream (default: wavlm_base_plus)
#   --gpu IDS          CUDA_VISIBLE_DEVICES value (default: 0)
#   --exp-name NAME    experiment name (default: <upstream>_asv)
#   --stage STAGE      all | train | evaluate (default: all)
#   --extra-override S extra s3prl -o overrides, appended verbatim after ',,' (default: none)
#   --dry-run          print commands without executing
# Env overrides:
#   S3PRL_ROOT           s3prl checkout (default: <repo>/external/s3prl)
#   SSL_BENCH_CONDA_ENV  conda env name/path (default: ssl-bench-s3prl)
#
# Downstream recipe: -d sv_voxceleb1 (default config.yaml). Data-root override
#   key: config.downstream_expert.datarc.file_path.
# Metric: EER (%) — test_expdir.sh scores states-{20000..200000}.ckpt on the
#   voxceleb1_test_v2 trials and prints "The best checkpoint achieves EER <val>";
#   lower is better. Per-checkpoint results land in result/downstream/<exp>/report.txt.
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

[[ -n "$DATA_ROOT" ]] || { echo "ERROR: --data-root is required (VoxCeleb1 root; see docs/superb.md#asv)" >&2; exit 2; }
case "$STAGE" in all|train|evaluate) ;; *) echo "ERROR: --stage must be all|train|evaluate" >&2; exit 2;; esac
EXP_NAME="${EXP_NAME:-${UPSTREAM}_asv}"
S3PRL_ROOT="${S3PRL_ROOT:-$REPO_ROOT/external/s3prl}"
[[ -d "$S3PRL_ROOT/s3prl" ]] || { echo "ERROR: s3prl not found at $S3PRL_ROOT (set S3PRL_ROOT or git submodule update --init)" >&2; exit 2; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SSL_BENCH_CONDA_ENV:-ssl-bench-s3prl}"
export CUDA_VISIBLE_DEVICES="$GPU"

LOG_DIR="$REPO_ROOT/logs/superb/asv"; mkdir -p "$LOG_DIR"
TRAIN_LOG="$LOG_DIR/${EXP_NAME}_train.log"
EVAL_LOG="$LOG_DIR/${EXP_NAME}_eval.log"
EXPDIR="result/downstream/${EXP_NAME}"
cd "$S3PRL_ROOT/s3prl"
run() { echo "+ $*"; [[ $DRY_RUN -eq 1 ]] || "$@"; }

# Data-root override key verified in downstream/sv_voxceleb1/config.yaml
# (config.downstream_expert.datarc.file_path). Extra overrides appended verbatim.
OVERRIDE="config.downstream_expert.datarc.file_path=${DATA_ROOT}"
[[ -n "$EXTRA_O" ]] && OVERRIDE="${OVERRIDE},,${EXTRA_O}"

# ---- Train (frozen upstream, learnable weighted-sum; no -f/-l per SUPERB) ----
# First run filters utts < 2 s and builds a cache in downstream/sv_voxceleb1/cache_wav_paths.
if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
  run bash -c "set -o pipefail; python3 run_downstream.py -m train -u '${UPSTREAM}' -d sv_voxceleb1 -n '${EXP_NAME}' -o '${OVERRIDE}' 2>&1 | tee '${TRAIN_LOG}'"
fi

# ---- Evaluate: test_expdir.sh loops states-*.ckpt, scores the test trials, ----
# ---- greps test-EER, and prints the best EER across checkpoints. ----
if [[ "$STAGE" == "all" || "$STAGE" == "evaluate" ]]; then
  if [[ $DRY_RUN -eq 0 ]]; then
    ls "$EXPDIR"/states-*.ckpt >/dev/null 2>&1 || { echo "ERROR: no states-*.ckpt under $(pwd)/$EXPDIR (run the train stage first)" >&2; exit 2; }
  fi
  run bash -c "set -o pipefail; ./downstream/sv_voxceleb1/test_expdir.sh '${EXPDIR}' '${DATA_ROOT}' 2>&1 | tee '${EVAL_LOG}'"
  if [[ $DRY_RUN -eq 0 ]]; then
    # "The best checkpoint achieves EER <val>" printed by test_expdir.sh.
    # EER is reported as a fraction by sv_voxceleb1/expert.py; x100 = EER %.
    EER="$(grep -oE 'The best checkpoint achieves EER [0-9.]+([eE][+-]?[0-9]+)?' "$EVAL_LOG" | tail -1 | awk '{print $NF}')"
    [[ -n "$EER" ]] || { echo "ERROR: could not parse best EER from $EVAL_LOG (see $EXPDIR/report.txt)" >&2; exit 1; }
    echo "RESULT superb asv eer=${EER}"
  fi
fi
