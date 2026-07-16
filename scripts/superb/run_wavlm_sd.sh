#!/usr/bin/env bash
# =============================================================================
# SUPERB SD — Speaker Diarization (Libri2Mix) · upstream: WavLM (s3prl)
# Trains a 2-speaker diarization head on frozen WavLM features, infers RTTM
# predictions from the dev-best checkpoint, and scores DER with dscore.
# Generic local/conda execution — no scheduler assumptions.
#
# Data are GENERATED (Libri2Mix from LibriSpeech + WHAM! noise) — see the 'prep'
# stage and docs/superb.md#sd. The diarization recipe reads the generated data
# from downstream/diarization/data/{train,dev,test} by default.
#
# Usage:
#   bash scripts/superb/run_wavlm_sd.sh [options]
# Options:
#   --data-root PATH   dir holding train/ dev/ test/ diarization data
#                      (default: downstream/diarization/data, i.e. where the
#                      prep step writes). Overrides loaderrc.{train,dev,test}_dir.
#   --dscore-dir PATH  path to a cloned https://github.com/ftshijt/dscore
#                      (REQUIRED for --stage score/all). The repo's score.sh
#                      hardcodes this path; we patch a COPY (never the clone).
#   --frame-shift N    override config.downstream_expert.datarc.frame_shift
#                      (default: unset = auto-match upstream; set 160 to shorten).
#   --upstream NAME    s3prl upstream (default: wavlm_base_plus)
#   --gpu IDS          CUDA_VISIBLE_DEVICES value (default: 0)
#   --exp-name NAME    experiment name (default: <upstream>_sd)
#   --stage STAGE      all | prep | train | infer | score (default: all = train+infer+score)
#   --extra-override S extra s3prl -o overrides, appended verbatim after ',,' (default: none)
#   --dry-run          print commands without executing
# Env overrides:
#   S3PRL_ROOT           s3prl checkout (default: <repo>/external/s3prl)
#   SSL_BENCH_CONDA_ENV  conda env name/path (default: ssl-bench-s3prl)
#
# Downstream recipe: -d diarization (default config.yaml). Data-root keys:
#   config.downstream_expert.loaderrc.{train,dev,test}_dir.
# Metric: DER (%) — score.sh sweeps median{1,11}×threshold{0.3..0.7}, runs dscore,
#   and the lowest DER (best config) is reported; lower is better. Checkpoint for
#   inference is best-states-dev.ckpt (NOT dev-best.ckpt).
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UPSTREAM="wavlm_base_plus"; GPU="${CUDA_VISIBLE_DEVICES:-0}"; DATA_ROOT="downstream/diarization/data"
EXP_NAME=""; STAGE="all"; EXTRA_O=""; DRY_RUN=0; DSCORE_DIR=""; FRAME_SHIFT=""

while [[ $# -gt 0 ]]; do case "$1" in
  --data-root) DATA_ROOT="$2"; shift 2;;  --upstream) UPSTREAM="$2"; shift 2;;
  --gpu) GPU="$2"; shift 2;;              --exp-name) EXP_NAME="$2"; shift 2;;
  --stage) STAGE="$2"; shift 2;;          --extra-override) EXTRA_O="$2"; shift 2;;
  --dscore-dir) DSCORE_DIR="$2"; shift 2;; --frame-shift) FRAME_SHIFT="$2"; shift 2;;
  --dry-run) DRY_RUN=1; shift;;
  *) echo "unknown option: $1" >&2; exit 2;; esac; done

case "$STAGE" in all|prep|train|infer|score) ;; *) echo "ERROR: --stage must be all|prep|train|infer|score" >&2; exit 2;; esac
EXP_NAME="${EXP_NAME:-${UPSTREAM}_sd}"
S3PRL_ROOT="${S3PRL_ROOT:-$REPO_ROOT/external/s3prl}"
[[ -d "$S3PRL_ROOT/s3prl" ]] || { echo "ERROR: s3prl not found at $S3PRL_ROOT (set S3PRL_ROOT or git submodule update --init)" >&2; exit 2; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SSL_BENCH_CONDA_ENV:-ssl-bench-s3prl}"
export CUDA_VISIBLE_DEVICES="$GPU"

LOG_DIR="$REPO_ROOT/logs/superb/sd"; mkdir -p "$LOG_DIR"
TRAIN_LOG="$LOG_DIR/${EXP_NAME}_train.log"
INFER_LOG="$LOG_DIR/${EXP_NAME}_infer.log"
SCORE_LOG="$LOG_DIR/${EXP_NAME}_score.log"
EXPDIR="result/downstream/${EXP_NAME}"
cd "$S3PRL_ROOT/s3prl"
run() { echo "+ $*"; [[ $DRY_RUN -eq 1 ]] || "$@"; }

# loaderrc overrides from --data-root, plus optional frame_shift + extra.
OVERRIDE="config.downstream_expert.loaderrc.train_dir=${DATA_ROOT}/train,,config.downstream_expert.loaderrc.dev_dir=${DATA_ROOT}/dev,,config.downstream_expert.loaderrc.test_dir=${DATA_ROOT}/test"
[[ -n "$FRAME_SHIFT" ]] && OVERRIDE="${OVERRIDE},,config.downstream_expert.datarc.frame_shift=${FRAME_SHIFT}"
[[ -n "$EXTRA_O" ]] && OVERRIDE="${OVERRIDE},,${EXTRA_O}"

# ---- prep: print the (heavy, external) Libri2Mix generation pointers only ----
do_prep() {
  cat <<'PREP'
[SD data prep — run manually; needs LibriSpeech + WHAM! noise and a WRITABLE storage dir]
  git clone https://github.com/s3prl/LibriMix.git && cd LibriMix
  # Edit the LibriSpeech / WHAM! paths inside generate_librimix_sd.sh to your local copies, then:
  bash generate_librimix_sd.sh <WRITABLE_STORAGE_DIR>
  # Convert the generated metadata into the diarization data layout the recipe reads:
  python3 scripts/prepare_diarization.py \
      --target_dir <S3PRL>/s3prl/downstream/diarization/data \
      --source_dir <WRITABLE_STORAGE_DIR>/Libri2Mix/wav16k/max/metadata
  # Result: downstream/diarization/data/{train,dev,test} (the default --data-root).
PREP
}

# ---- train ----
do_train() {
  run bash -c "set -o pipefail; python3 run_downstream.py -m train -u '${UPSTREAM}' -d diarization -n '${EXP_NAME}' -o '${OVERRIDE}' 2>&1 | tee '${TRAIN_LOG}'"
}

# ---- infer: RTTM predictions from best-states-dev.ckpt (test split → *.h5) ----
do_infer() {
  local ckpt="${EXPDIR}/best-states-dev.ckpt"
  [[ $DRY_RUN -eq 1 || -f "$ckpt" ]] || { echo "ERROR: checkpoint not found: $(pwd)/$ckpt (run the train stage first)" >&2; exit 2; }
  run bash -c "set -o pipefail; python3 run_downstream.py -m evaluate -t test -e '${ckpt}' 2>&1 | tee '${INFER_LOG}'"
}

# ---- score: patch a COPY of score.sh with --dscore-dir, run, extract DER ----
do_score() {
  [[ -n "$DSCORE_DIR" ]] || { echo "ERROR: --dscore-dir is required for the score stage (clone https://github.com/ftshijt/dscore)" >&2; exit 2; }
  local patched="$LOG_DIR/score_${EXP_NAME}.sh" testset="${DATA_ROOT}/test"
  # Never edit the s3prl clone: sed a patched copy into LOG_DIR (rule: repo/clone stay clean).
  run bash -c "sed 's|^dscore_dir=.*|dscore_dir=${DSCORE_DIR}|' downstream/diarization/score.sh > '${patched}'"
  # stdout (the sorted OVERALL rows) → SCORE_LOG; set -x trace + dscore warnings → trace log.
  run bash -c "set -o pipefail; bash '${patched}' '${EXPDIR}' '${testset}' 2>'${LOG_DIR}/${EXP_NAME}_score_trace.log' | tee '${SCORE_LOG}'"
  if [[ $DRY_RUN -eq 0 ]]; then
    # DER is field 4 of the '<file>:*** OVERALL ***  DER JER ...' line (grep's filename
    # prefix + '*** OVERALL ***' shifts DER to column 4 — the column score.sh sorts on).
    DER="$(grep -h 'OVERALL' "$SCORE_LOG" | awk 'NF>=4{print $4}' | sort -n | head -1)"
    [[ -n "$DER" ]] || { echo "ERROR: could not parse DER from $SCORE_LOG (inspect it + the dscore output)" >&2; exit 1; }
    echo "RESULT superb sd der=${DER} (lowest across median/threshold sweep)"
  fi
}

case "$STAGE" in
  prep)  do_prep ;;
  train) do_train ;;
  infer) do_infer ;;
  score) do_score ;;
  all)   do_train; do_infer; do_score ;;
esac
