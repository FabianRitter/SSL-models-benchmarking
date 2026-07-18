#!/usr/bin/env bash
# =============================================================================
# ML-SUPERB 2.0 (Interspeech-2024 Challenge baseline) — WavLM Base+ (frozen SSL)
# Wraps egs2/ml_superb2/asr1/run.sh: one config, one train/dev split, CTC.
# One command drives ESPnet stages 1-13 (data-prep -> train -> decode -> score).
# Generic local/conda execution — no scheduler assumptions.
#
# DATA: ~15.5 GB AUTO-DOWNLOADS from Hugging Face 'espnet/ml_superb_hf' during
#       stage-1 data prep (public, ungated, license per the HF repo). It caches
#       INTO the recipe dir (the recipe hard-codes cache_dir="."); budget ~16 GB
#       of free space there plus prepared data/.
# SCORING IS DEV-ONLY: the recipe ships only the public dev / dev_dialect splits;
#       the challenge TEST set is held out. Every number here is a DEV number —
#       always label it "dev", never as a leaderboard / test result.
#
# Usage:
#   bash scripts/ml_superb/run_wavlm_mlsuperb2.sh [options]
# Options:
#   --asr-config PATH   Training YAML (default: conf/tuning/train_wavlm_baseline.yaml).
#   --stage N           ESPnet start stage (default: 1; stage 1 does the HF download).
#   --stop-stage N      ESPnet stop stage  (default: 13).
#   --gpu IDS           CUDA_VISIBLE_DEVICES value (default: 0).
#   --asr-args "STR"    Extra args forwarded to ESPnet asr training (asr.sh --asr_args),
#                       e.g. --asr-args "--max_epoch 2 --num_iters_per_epoch 2000" to
#                       cap a SMOKE run. NOTE: a non-empty value changes the exp/ tag
#                       (asr.sh appends a sanitised copy of the args to asr_tag).
#   --exp-suffix S      Provenance tag appended to the wrapper log filename.
#   --dry-run           Print the exact recipe command without executing.
#   -- ARGS...          Everything after a literal '--' is forwarded verbatim to the
#                       recipe run.sh -> asr.sh (last-wins), e.g. '-- --inference_nj 8'
#                       to cap single-GPU decode parallelism (recipe default is 64,
#                       which OOMs one GPU with run.pl local backend).
# Env overrides:
#   ESPNET_ROOT           ESPnet checkout (default: <repo>/external/espnet).
#   SSL_BENCH_ESPNET_ENV  conda env name/path (default: ssl-bench-espnet).
#
# Requires (in the env): espnet2, espnet-fork s3prl, sox, sclite, datasets (HF), jiwer.
# WavLM Base+ ckpt (~0.4 GB) auto-downloads to <recipe>/hub on first run.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ASR_CONFIG="conf/tuning/train_wavlm_baseline.yaml"
STAGE=1; STOP_STAGE=13; GPU="${CUDA_VISIBLE_DEVICES:-0}"; EXP_SUFFIX=""; DRY_RUN=0; ASR_ARGS=""
PASSTHRU=()
while [[ $# -gt 0 ]]; do case "$1" in
  --asr-config) ASR_CONFIG="$2"; shift 2;;
  --stage)      STAGE="$2"; shift 2;;
  --stop-stage) STOP_STAGE="$2"; shift 2;;
  --gpu)        GPU="$2"; shift 2;;
  --asr-args)   ASR_ARGS="$2"; shift 2;;
  --exp-suffix) EXP_SUFFIX="$2"; shift 2;;
  --)           shift; PASSTHRU=("$@"); break;;
  -h|--help)    sed -n '2,37p' "$0"; exit 0;;
  --dry-run)    DRY_RUN=1; shift;;
  *) echo "unknown option: $1" >&2; exit 2;; esac; done

# ---- resolve recipe dir ----
ESPNET_ROOT="${ESPNET_ROOT:-$REPO_ROOT/external/espnet}"
RECIPE_DIR="$ESPNET_ROOT/egs2/ml_superb2/asr1"
[[ -d "$RECIPE_DIR" ]] || { echo "ERROR: recipe not found at $RECIPE_DIR (set ESPNET_ROOT or 'git submodule update --init')" >&2; exit 2; }

# ---- validate config presence (skip on dry-run; submodule may predate the WavLM re-pin) ----
if [[ $DRY_RUN -eq 0 && ! -f "$RECIPE_DIR/$ASR_CONFIG" ]]; then
  echo "ERROR: asr config not found: $RECIPE_DIR/$ASR_CONFIG" >&2
  echo "       (WavLM config lives on espnet branch 'ml-superb-wavlm'; pin external/espnet to it)" >&2
  exit 2
fi

# ---- activate env (only for real runs; dry-run stays dependency-free) ----
if [[ $DRY_RUN -eq 0 ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${SSL_BENCH_ESPNET_ENV:-ssl-bench-espnet}"
fi
export CUDA_VISIBLE_DEVICES="$GPU"

LOG_DIR="$REPO_ROOT/logs/ml_superb"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/mlsuperb2_baseline${EXP_SUFFIX:+_$EXP_SUFFIX}.log"

echo "=============================================================================="
echo " ML-SUPERB 2.0 (challenge baseline, DEV-set scoring) | WavLM Base+ frozen"
echo " recipe = $RECIPE_DIR"
echo " config = $ASR_CONFIG"
echo " data   = auto-download HF espnet/ml_superb_hf (~15.5 GB) into the recipe dir at stage 1"
echo " env    = ${SSL_BENCH_ESPNET_ENV:-ssl-bench-espnet}   gpu=$GPU   stages ${STAGE}..${STOP_STAGE}"
echo " log    = $LOG"
echo "=============================================================================="

# run.sh forwards "$@" to asr.sh; a later --asr_config overrides run.sh's built-in
# default (conf/train_asr.yaml = the MMS-1B baseline). ESPnet parse_options is
# last-wins, so our WavLM config takes effect. --stage/--stop_stage also flow through.
# --asr_args (if set) is forwarded to asr.sh, which appends it to the asr_train
# invocation (used to cap epochs for a SMOKE run).
CMD=( ./run.sh --asr_config "$ASR_CONFIG" --stage "$STAGE" --stop_stage "$STOP_STAGE" )
[[ -n "$ASR_ARGS" ]] && CMD+=( --asr_args "$ASR_ARGS" )
[[ ${#PASSTHRU[@]} -gt 0 ]] && CMD+=( "${PASSTHRU[@]}" )

echo "+ (cd $RECIPE_DIR && ${CMD[*]})"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "RESULT ml_superb2 baseline dry-run=ok"
  exit 0
fi

( cd "$RECIPE_DIR" && "${CMD[@]}" ) 2>&1 | tee "$LOG"

echo "------------------------------------------------------------------------------"
echo "Challenge metrics (DEV): $RECIPE_DIR/exp/<asr_tag>/challenge_results.md"
echo "  columns: Standard CER | Standard LID | Worst-15 CER | CER StD | Dialect CER | Dialect LID"
echo "  re-score: (cd $RECIPE_DIR && python local/score.py --exp_dir exp/<asr_tag>)"
echo "RESULT ml_superb2 baseline log=$LOG"
