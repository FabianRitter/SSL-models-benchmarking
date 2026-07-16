#!/usr/bin/env bash
# =============================================================================
# SUPERB-SG SS — Source Separation (Libri2Mix) · upstream: WavLM (s3prl)
# Trains a SepRNN mask head on frozen WavLM features to separate a 2-speaker
# mixture (mix_clean → s1,s2), then evaluates SI-SDRi on the test set.
# Uses the v1 'separation_stft' recipe (SUPERB-comparable; separation_stft2 is an
# improved recipe that is NOT comparable). Generic local/conda — no scheduler.
#
# Data are GENERATED (Libri2Mix 16 kHz 'min') — see the 'prep' stage and
# docs/superb_sg.md#ss. The recipe reads
# downstream/separation_stft/datasets/Libri2Mix/wav16k/min/{train-100,dev_1000,test}.
#
# Usage:
#   bash scripts/superb_sg/run_wavlm_ss.sh [options]
# Options:
#   --data-root PATH   dir holding train-100/ dev_1000/ test/ subdirs
#                      (default: downstream/separation_stft/datasets/Libri2Mix/wav16k/min).
#                      Overrides loaderrc.{train,dev,test}_dir.
#   --upstream NAME    s3prl upstream (default: wavlm_base_plus)
#   --gpu IDS          CUDA_VISIBLE_DEVICES value (default: 0)
#   --exp-name NAME    experiment name (default: <upstream>_ss)
#   --stage STAGE      all | prep | train | evaluate (default: all = train+evaluate)
#   --extra-override S extra s3prl -o overrides, appended verbatim after ',,' (default: none)
#   --dry-run          print commands without executing
# Env overrides:
#   S3PRL_ROOT           s3prl checkout (default: <repo>/external/s3prl)
#   SSL_BENCH_CONDA_ENV  conda env name/path (default: ssl-bench-s3prl)
#
# Downstream recipe: -d separation_stft -c downstream/separation_stft/configs/cfg.yaml
#   Data-root keys: config.downstream_expert.loaderrc.{train,dev,test}_dir. Note the
#   train/dev/test subdir names are train-100 / dev_1000 / test.
# Metric: SI-SDRi (dB, ↑) — written to result/downstream/<exp>/test_metrics.txt
#   (stoi/pesq also written). Checkpoint = best-states-dev.ckpt.
# Requires asteroid==0.4.4 in the env.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UPSTREAM="wavlm_base_plus"; GPU="${CUDA_VISIBLE_DEVICES:-0}"
DATA_ROOT="downstream/separation_stft/datasets/Libri2Mix/wav16k/min"
EXP_NAME=""; STAGE="all"; EXTRA_O=""; DRY_RUN=0
SS_CONFIG="downstream/separation_stft/configs/cfg.yaml"
# subdir names differ from train/dev/test — verified in cfg.yaml.
TRAIN_SUB="train-100"; DEV_SUB="dev_1000"; TEST_SUB="test"

while [[ $# -gt 0 ]]; do case "$1" in
  --data-root) DATA_ROOT="$2"; shift 2;;  --upstream) UPSTREAM="$2"; shift 2;;
  --gpu) GPU="$2"; shift 2;;              --exp-name) EXP_NAME="$2"; shift 2;;
  --stage) STAGE="$2"; shift 2;;          --extra-override) EXTRA_O="$2"; shift 2;;
  --dry-run) DRY_RUN=1; shift;;
  *) echo "unknown option: $1" >&2; exit 2;; esac; done

case "$STAGE" in all|prep|train|evaluate) ;; *) echo "ERROR: --stage must be all|prep|train|evaluate" >&2; exit 2;; esac
EXP_NAME="${EXP_NAME:-${UPSTREAM}_ss}"
S3PRL_ROOT="${S3PRL_ROOT:-$REPO_ROOT/external/s3prl}"
[[ -d "$S3PRL_ROOT/s3prl" ]] || { echo "ERROR: s3prl not found at $S3PRL_ROOT (set S3PRL_ROOT or git submodule update --init)" >&2; exit 2; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SSL_BENCH_CONDA_ENV:-ssl-bench-s3prl}"
export CUDA_VISIBLE_DEVICES="$GPU"

LOG_DIR="$REPO_ROOT/logs/superb_sg/ss"; mkdir -p "$LOG_DIR"
TRAIN_LOG="$LOG_DIR/${EXP_NAME}_train.log"
EVAL_LOG="$LOG_DIR/${EXP_NAME}_eval.log"
EXPDIR="result/downstream/${EXP_NAME}"
cd "$S3PRL_ROOT/s3prl"
run() { echo "+ $*"; [[ $DRY_RUN -eq 1 ]] || "$@"; }

OVERRIDE="config.downstream_expert.loaderrc.train_dir=${DATA_ROOT}/${TRAIN_SUB},,config.downstream_expert.loaderrc.dev_dir=${DATA_ROOT}/${DEV_SUB},,config.downstream_expert.loaderrc.test_dir=${DATA_ROOT}/${TEST_SUB}"
[[ -n "$EXTRA_O" ]] && OVERRIDE="${OVERRIDE},,${EXTRA_O}"

do_prep() {
  cat <<'PREP'
[SS data prep — run manually; needs LibriSpeech + WHAM! noise and a WRITABLE storage dir]
  git clone https://github.com/s3prl/LibriMix.git && cd LibriMix
  # Edit the LibriSpeech / WHAM! paths inside generate_librimix_ss.sh to your local copies, then:
  ./generate_librimix_ss.sh <WRITABLE_STORAGE_DIR>
  cd <S3PRL>/s3prl
  # Build the recipe's scp layout (run once per partition):
  for part in train-100 dev test; do
    python3 downstream/separation_stft/scripts/LibriMix/data_prepare.py \
        --part $part --sample_rate 16k --mode min \
        <WRITABLE_STORAGE_DIR>/Libri2Mix downstream/separation_stft/datasets/Libri2Mix
  done
  # Subsample dev -> dev_1000 (the dev split the config expects):
  python3 downstream/separation_stft/scripts/LibriMix/subsample.py \
      downstream/separation_stft/datasets/Libri2Mix/wav16k/min/dev \
      downstream/separation_stft/datasets/Libri2Mix/wav16k/min/dev_1000 --sample 1000
  # Result: .../Libri2Mix/wav16k/min/{train-100,dev_1000,test} (default --data-root).
PREP
}

do_train() {
  run bash -c "set -o pipefail; python3 run_downstream.py -m train -u '${UPSTREAM}' -d separation_stft -c '${SS_CONFIG}' -n '${EXP_NAME}' -o '${OVERRIDE}' 2>&1 | tee '${TRAIN_LOG}'"
}

do_evaluate() {
  local ckpt="${EXPDIR}/best-states-dev.ckpt"
  [[ $DRY_RUN -eq 1 || -f "$ckpt" ]] || { echo "ERROR: checkpoint not found: $(pwd)/$ckpt (run the train stage first)" >&2; exit 2; }
  run bash -c "set -o pipefail; python3 run_downstream.py -m evaluate -t test -e '${ckpt}' 2>&1 | tee '${EVAL_LOG}'"
  if [[ $DRY_RUN -eq 0 ]]; then
    local met="${EXPDIR}/test_metrics.txt"
    [[ -f "$met" ]] || { echo "ERROR: $met not written by evaluate — check $EVAL_LOG" >&2; exit 1; }
    local sisdr stoi pesq
    sisdr="$(awk '$1=="si_sdr"{print $2}' "$met" | tail -1)"
    stoi="$(awk '$1=="stoi"{print $2}'  "$met" | tail -1)"
    pesq="$(awk '$1=="pesq"{print $2}'  "$met" | tail -1)"
    echo "RESULT superb_sg ss si_sdri=${sisdr:-NA} (stoi=${stoi:-NA} pesq=${pesq:-NA})"
  fi
}

case "$STAGE" in
  prep)     do_prep ;;
  train)    do_train ;;
  evaluate) do_evaluate ;;
  all)      do_train; do_evaluate ;;
esac
