#!/usr/bin/env bash
# =============================================================================
# SUPERB-SG SE — Speech Enhancement (Voicebank-DEMAND) · upstream: WavLM (s3prl)
# Trains a SepRNN masking head on frozen WavLM features to map noisy speech to
# clean, then evaluates PESQ / STOI / SI-SDRi on the test set.
# Uses the v1 'enhancement_stft' recipe (SUPERB-comparable; enhancement_stft2 is
# an improved recipe that is NOT comparable). Generic local/conda — no scheduler.
#
# Data are prepared into an in-tree Kaldi-scp layout — see the 'prep' stage and
# docs/superb_sg.md#se. The recipe reads
# downstream/enhancement_stft/datasets/voicebank/wav16k/{train,dev,test} by default.
#
# Usage:
#   bash scripts/superb_sg/run_wavlm_se.sh [options]
# Options:
#   --data-root PATH   dir holding train/ dev/ test/ scp subdirs
#                      (default: downstream/enhancement_stft/datasets/voicebank/wav16k).
#                      Overrides loaderrc.{train,dev,test}_dir.
#   --upstream NAME    s3prl upstream (default: wavlm_base_plus)
#   --gpu IDS          CUDA_VISIBLE_DEVICES value (default: 0)
#   --exp-name NAME    experiment name (default: <upstream>_se)
#   --stage STAGE      all | prep | train | evaluate (default: all = train+evaluate)
#   --extra-override S extra s3prl -o overrides, appended verbatim after ',,' (default: none)
#   --dry-run          print commands without executing
# Env overrides:
#   S3PRL_ROOT           s3prl checkout (default: <repo>/external/s3prl)
#   SSL_BENCH_CONDA_ENV  conda env name/path (default: ssl-bench-s3prl)
#
# Downstream recipe: -d enhancement_stft -c downstream/enhancement_stft/configs/cfg_voicebank.yaml
#   Data-root keys: config.downstream_expert.loaderrc.{train,dev,test}_dir.
# Metrics: PESQ (↑), STOI (↑), SI-SDRi (dB, ↑) — written to
#   result/downstream/<exp>/test_metrics.txt. Checkpoint = best-states-dev.ckpt.
# Requires asteroid==0.4.4 (pulls pesq, pystoi) in the env.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UPSTREAM="wavlm_base_plus"; GPU="${CUDA_VISIBLE_DEVICES:-0}"
DATA_ROOT="downstream/enhancement_stft/datasets/voicebank/wav16k"
EXP_NAME=""; STAGE="all"; EXTRA_O=""; DRY_RUN=0
SE_CONFIG="downstream/enhancement_stft/configs/cfg_voicebank.yaml"

while [[ $# -gt 0 ]]; do case "$1" in
  --data-root) DATA_ROOT="$2"; shift 2;;  --upstream) UPSTREAM="$2"; shift 2;;
  --gpu) GPU="$2"; shift 2;;              --exp-name) EXP_NAME="$2"; shift 2;;
  --stage) STAGE="$2"; shift 2;;          --extra-override) EXTRA_O="$2"; shift 2;;
  --dry-run) DRY_RUN=1; shift;;
  *) echo "unknown option: $1" >&2; exit 2;; esac; done

case "$STAGE" in all|prep|train|evaluate) ;; *) echo "ERROR: --stage must be all|prep|train|evaluate" >&2; exit 2;; esac
EXP_NAME="${EXP_NAME:-${UPSTREAM}_se}"
S3PRL_ROOT="${S3PRL_ROOT:-$REPO_ROOT/external/s3prl}"
[[ -d "$S3PRL_ROOT/s3prl" ]] || { echo "ERROR: s3prl not found at $S3PRL_ROOT (set S3PRL_ROOT or git submodule update --init)" >&2; exit 2; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SSL_BENCH_CONDA_ENV:-ssl-bench-s3prl}"
export CUDA_VISIBLE_DEVICES="$GPU"

LOG_DIR="$REPO_ROOT/logs/superb_sg/se"; mkdir -p "$LOG_DIR"
TRAIN_LOG="$LOG_DIR/${EXP_NAME}_train.log"
EVAL_LOG="$LOG_DIR/${EXP_NAME}_eval.log"
EXPDIR="result/downstream/${EXP_NAME}"
cd "$S3PRL_ROOT/s3prl"
run() { echo "+ $*"; [[ $DRY_RUN -eq 1 ]] || "$@"; }

OVERRIDE="config.downstream_expert.loaderrc.train_dir=${DATA_ROOT}/train,,config.downstream_expert.loaderrc.dev_dir=${DATA_ROOT}/dev,,config.downstream_expert.loaderrc.test_dir=${DATA_ROOT}/test"
[[ -n "$EXTRA_O" ]] && OVERRIDE="${OVERRIDE},,${EXTRA_O}"

do_prep() {
  cat <<'PREP'
[SE data prep — run manually; needs Voicebank-DEMAND 16 kHz]
  # Get Voicebank-DEMAND 16 kHz (Edinburgh DataShare 10283/2791; the 16 kHz
  # clean_/noisy_ trainset_28spk_wav_16k + testset_wav_16k dirs), call its parent <VB_SRC>.
  # Prepare the Kaldi-scp layout the recipe reads (run once per partition):
  for part in train dev test; do
    python3 downstream/enhancement_stft/scripts/Voicebank/data_prepare.py \
        <VB_SRC> downstream/enhancement_stft/datasets/voicebank --part $part --sample_rate 16k
  done
  # Result: downstream/enhancement_stft/datasets/voicebank/wav16k/{train,dev,test} (default --data-root).
PREP
}

do_train() {
  run bash -c "set -o pipefail; python3 run_downstream.py -m train -u '${UPSTREAM}' -d enhancement_stft -c '${SE_CONFIG}' -n '${EXP_NAME}' -o '${OVERRIDE}' 2>&1 | tee '${TRAIN_LOG}'"
}

do_evaluate() {
  local ckpt="${EXPDIR}/best-states-dev.ckpt"
  [[ $DRY_RUN -eq 1 || -f "$ckpt" ]] || { echo "ERROR: checkpoint not found: $(pwd)/$ckpt (run the train stage first)" >&2; exit 2; }
  run bash -c "set -o pipefail; python3 run_downstream.py -m evaluate -t test -e '${ckpt}' 2>&1 | tee '${EVAL_LOG}'"
  if [[ $DRY_RUN -eq 0 ]]; then
    # test_metrics.txt holds one '<metric> <value>' line per metric (si_sdr is SI-SDRi).
    local met="${EXPDIR}/test_metrics.txt"
    [[ -f "$met" ]] || { echo "ERROR: $met not written by evaluate — check $EVAL_LOG" >&2; exit 1; }
    local pesq stoi sisdr
    pesq="$(awk '$1=="pesq"{print $2}'  "$met" | tail -1)"
    stoi="$(awk '$1=="stoi"{print $2}'  "$met" | tail -1)"
    sisdr="$(awk '$1=="si_sdr"{print $2}' "$met" | tail -1)"
    echo "RESULT superb_sg se pesq=${pesq:-NA} stoi=${stoi:-NA} si_sdri=${sisdr:-NA}"
  fi
}

case "$STAGE" in
  prep)     do_prep ;;
  train)    do_train ;;
  evaluate) do_evaluate ;;
  all)      do_train; do_evaluate ;;
esac
