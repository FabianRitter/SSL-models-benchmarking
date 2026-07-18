#!/usr/bin/env bash
# =============================================================================
# ML-SUPERB 1.0 — WavLM Base+ (frozen SSL, s3prl frontend) · ESPnet recipe wrapper
# Wraps egs2/ml_superb/asr1 {run.sh, run_multi.sh} for the four core WavLM tracks:
#   monolingual ASR (10min / 1h, one language) and multilingual ASR (10min / 1h).
# One command drives ESPnet stages 1-13 (data-prep -> train -> decode -> score).
# Generic local/conda execution — no scheduler assumptions.
#
# Usage:
#   bash scripts/ml_superb/run_wavlm_mlsuperb1.sh --track TRACK --data-root PATH [options]
#
# Required:
#   --track TRACK       mono10min | mono1h | multi10min | multi1h
#   --data-root PATH    Unpacked ML-SUPERB 1.0 corpus dir (the 7th-version
#                       'seventh_version_unpacked' dir that DIRECTLY contains the
#                       source folders: mls/ commonvoice/ fleurs/ nchlt/ ...).
#                       The wrapper writes this into the recipe's db.sh (MLSUPERB=),
#                       which is the recipe's documented data mechanism.
#                       See doc_sections/ml_superb_overview.md (Data acquisition).
# Options:
#   --lang LANG         mono tracks only; one of
#                         eng1 eng2 eng3 fra1 fra2 deu1 deu2 rus swa swe jpn cmn xty
#                       (default: xty — the smallest / cheapest smoke language).
#                       NOTE: cmn/jpn are scored as PER (word/g2p) by the recipe's
#                       ./run_mono.sh; this single-language path uses char tokens —
#                       use ./run_mono.sh directly for the official cmn/jpn number.
#   --asr-config PATH   Override the training YAML (default: the track's WavLM
#                       config under conf/tuning/). Path is relative to the recipe.
#   --stage N           ESPnet start stage (default: 1).
#   --stop-stage N      ESPnet stop stage  (default: 13).
#   --gpu IDS           CUDA_VISIBLE_DEVICES value (default: 0).
#   --exp-suffix S      Provenance tag appended to the wrapper log filename.
#                       (ESPnet derives the exp/ dir name from the config + track;
#                       it is not affected by this flag.)
#   --dry-run           Print the exact recipe command(s) without executing.
# Env overrides:
#   ESPNET_ROOT           ESPnet checkout (default: <repo>/external/espnet).
#   SSL_BENCH_ESPNET_ENV  conda env name/path (default: ssl-bench-espnet).
#
# Requires (in the env): espnet2, espnet-fork s3prl (frontend), sox, sclite.
# WavLM Base+ ckpt (~0.4 GB) auto-downloads to <recipe>/hub on first run.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TRACK=""; LANG_SEL=""; DATA_ROOT=""; ASR_CONFIG=""
STAGE=1; STOP_STAGE=13; GPU="${CUDA_VISIBLE_DEVICES:-0}"; EXP_SUFFIX=""; DRY_RUN=0
while [[ $# -gt 0 ]]; do case "$1" in
  --track)      TRACK="$2"; shift 2;;
  --lang)       LANG_SEL="$2"; shift 2;;
  --data-root)  DATA_ROOT="$2"; shift 2;;
  --asr-config) ASR_CONFIG="$2"; shift 2;;
  --stage)      STAGE="$2"; shift 2;;
  --stop-stage) STOP_STAGE="$2"; shift 2;;
  --gpu)        GPU="$2"; shift 2;;
  --exp-suffix) EXP_SUFFIX="$2"; shift 2;;
  -h|--help)    sed -n '2,44p' "$0"; exit 0;;
  --dry-run)    DRY_RUN=1; shift;;
  *) echo "unknown option: $1" >&2; exit 2;; esac; done

# ---- validate track -> (duration, mode) ----
case "$TRACK" in
  mono10min)  DURATION=10min; MODE=mono;;
  mono1h)     DURATION=1h;    MODE=mono;;
  multi10min) DURATION=10min; MODE=multi;;
  multi1h)    DURATION=1h;    MODE=multi;;
  "") echo "ERROR: --track is required (mono10min|mono1h|multi10min|multi1h)" >&2; exit 2;;
  *)  echo "ERROR: unknown --track '$TRACK' (mono10min|mono1h|multi10min|multi1h)" >&2; exit 2;;
esac
[[ -n "$DATA_ROOT" ]] || { echo "ERROR: --data-root is required (unpacked ML-SUPERB 1.0 corpus)" >&2; exit 2; }

# ---- default WavLM config per track ----
if [[ -z "$ASR_CONFIG" ]]; then
  case "$TRACK" in
    mono10min|mono1h) ASR_CONFIG="conf/tuning/train_asr_s3prl_wavlm_single.yaml";;
    multi10min)       ASR_CONFIG="conf/tuning/train_asr_s3prl_wavlm_10min.yaml";;
    multi1h)          ASR_CONFIG="conf/tuning/train_asr_s3prl_wavlm_1h.yaml";;
  esac
fi

# ---- mono: language selection + validation ----
VALID_LANGS="eng1 eng2 eng3 fra1 fra2 deu1 deu2 rus swa swe jpn cmn xty"
LANG_TAG=""; LANG_DESC="(all languages)"
if [[ "$MODE" == "mono" ]]; then
  LANG_SEL="${LANG_SEL:-xty}"
  case " $VALID_LANGS " in
    *" $LANG_SEL "*) : ;;
    *) echo "ERROR: --lang '$LANG_SEL' not one of: $VALID_LANGS" >&2; exit 2;;
  esac
  if [[ "$LANG_SEL" == "cmn" || "$LANG_SEL" == "jpn" ]]; then
    echo "WARNING: '$LANG_SEL' is scored as PER via word/g2p tokens (matching ./run_mono.sh)." >&2
    echo "         This needs the g2p deps installed in the env (jpn: pyopenjtalk — data.sh" >&2
    echo "         auto-installs it; cmn: pypinyin — 'pip install pypinyin'). Result lands in" >&2
    echo "         score_wer/ (PER), not score_cer/." >&2
  fi
  LANG_TAG="_${LANG_SEL}"; LANG_DESC="lang=$LANG_SEL"
fi

# ---- resolve recipe dir ----
ESPNET_ROOT="${ESPNET_ROOT:-$REPO_ROOT/external/espnet}"
RECIPE_DIR="$ESPNET_ROOT/egs2/ml_superb/asr1"
[[ -d "$RECIPE_DIR" ]] || { echo "ERROR: recipe not found at $RECIPE_DIR (set ESPNET_ROOT or 'git submodule update --init')" >&2; exit 2; }

# ---- validate config presence (skip on dry-run; submodule may predate the WavLM re-pin) ----
if [[ $DRY_RUN -eq 0 && ! -f "$RECIPE_DIR/$ASR_CONFIG" ]]; then
  echo "ERROR: asr config not found: $RECIPE_DIR/$ASR_CONFIG" >&2
  echo "       (WavLM configs live on espnet branch 'ml-superb-wavlm'; pin external/espnet to it)" >&2
  exit 2
fi

# ---- validate data-root (skip existence checks on dry-run) ----
if [[ $DRY_RUN -eq 0 ]]; then
  [[ -d "$DATA_ROOT" ]] || { echo "ERROR: --data-root does not exist: $DATA_ROOT" >&2; exit 3; }
  if [[ ! -d "$DATA_ROOT/mls" && ! -d "$DATA_ROOT/commonvoice" ]]; then
    echo "WARNING: $DATA_ROOT has no mls/ or commonvoice/ source dir — point --data-root at" >&2
    echo "         the unpacked 'seventh_version_unpacked' dir (see the overview doc)." >&2
  fi
fi

# ---- activate env (only for real runs; dry-run stays dependency-free) ----
if [[ $DRY_RUN -eq 0 ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${SSL_BENCH_ESPNET_ENV:-ssl-bench-espnet}"
fi
export CUDA_VISIBLE_DEVICES="$GPU"

LOG_DIR="$REPO_ROOT/logs/ml_superb"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/mlsuperb1_${TRACK}${LANG_TAG}${EXP_SUFFIX:+_$EXP_SUFFIX}.log"

# db.sh is a symlink to egs2/TEMPLATE/asr1/db.sh; resolve it so the symlink is preserved.
DB_LINK="$RECIPE_DIR/db.sh"
DB_REAL="$(readlink -f "$DB_LINK" 2>/dev/null || echo "$DB_LINK")"

echo "=============================================================================="
echo " ML-SUPERB 1.0 | track=$TRACK  dur=$DURATION  $LANG_DESC"
echo " recipe    = $RECIPE_DIR"
echo " config    = $ASR_CONFIG"
echo " data-root = $DATA_ROOT"
echo "             -> MLSUPERB= in $DB_REAL"
echo " env       = ${SSL_BENCH_ESPNET_ENV:-ssl-bench-espnet}   gpu=$GPU   stages ${STAGE}..${STOP_STAGE}"
echo " log       = $LOG"
echo "=============================================================================="

# ---- write the corpus path into db.sh (the recipe's documented data mechanism) ----
if [[ $DRY_RUN -eq 1 ]]; then
  echo "+ (dry-run) sed -i 's#^MLSUPERB=.*#MLSUPERB=$DATA_ROOT#' $DB_REAL"
else
  sed -i "s#^MLSUPERB=.*#MLSUPERB=${DATA_ROOT}#" "$DB_REAL"
  echo "+ $(grep -E '^MLSUPERB=' "$DB_REAL" | head -1)  ($DB_REAL)"
fi

# ---- build the recipe command ----
if [[ "$MODE" == "multi" ]]; then
  # Canonical multilingual ASR (README: --lid false --only_lid false).
  CMD=( ./run_multi.sh
        --stage "$STAGE" --stop_stage "$STOP_STAGE"
        --duration "$DURATION" --lid false --only_lid false
        --asr_config "$ASR_CONFIG" )
else
  # Monolingual single-language.
  # IMPORTANT: we do NOT use './run.sh --single_lang ...'. The recipe's run.sh EXECUTES
  # ('./utils/parse_options.sh') instead of SOURCING ('. utils/parse_options.sh') the option
  # parser, so run.sh IGNORES all CLI flags and always runs its LID-multilingual-fbank
  # defaults (verified empirically). run_multi.sh sources it (so the multi branch above is
  # flag-driven and correct), but run_mono.sh also executes it and instead drives asr.sh
  # directly, per-language, inside a loop. We therefore replicate run_mono.sh's exact
  # single-language ./asr.sh invocation here (source-faithful; NO espnet edit).
  #  - dev/test sets are ALWAYS the fixed 10min utterances (run_mono hardcodes dev_10min/
  #    test_10min even for the 1h track); only train_set carries the duration.
  #  - score_type is 'monolingual'; CPU decode (gpu_inference=false) as in run_mono.
  if [[ "$LANG_SEL" == "cmn" || "$LANG_SEL" == "jpn" ]]; then TOKEN_TYPE=word; else TOKEN_TYPE=char; fi
  MONO_TRAIN="train_${DURATION}_${LANG_SEL}"
  MONO_DEV="dev_10min_${LANG_SEL}"
  MONO_TEST="dev_10min_${LANG_SEL} test_10min_${LANG_SEL}"
  MONO_TAG="$(basename "$ASR_CONFIG" .yaml)_${LANG_SEL}_${DURATION}"
  MONO_DATA_OPTS="--duration ${DURATION} --lid false --multilingual false --single_lang ${LANG_SEL}"
  CMD=( ./asr.sh
        --ngpu 1
        --stage "$STAGE" --stop_stage "$STOP_STAGE"
        --nj 16 --inference_nj 16 --gpu_inference false
        --lang "$LANG_SEL"
        --inference_asr_model valid.loss.ave.pth
        --local_data_opts "$MONO_DATA_OPTS"
        --use_lm false
        --token_type "$TOKEN_TYPE"
        --feats_type raw
        --feats_normalize utterance_mvn
        --asr_config "$ASR_CONFIG"
        --inference_config conf/decode_asr.yaml
        --train_set "$MONO_TRAIN"
        --valid_set "$MONO_DEV"
        --test_sets "$MONO_TEST"
        --asr_tag "$MONO_TAG"
        --expdir exp
        --asr_stats_dir "exp/asr_stats_${LANG_SEL}_${DURATION}"
        --local_score_opts "false false monolingual" )
fi

echo "+ (cd $RECIPE_DIR && ${CMD[*]})"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "RESULT ml_superb1 $TRACK dry-run=ok"
  exit 0
fi

( cd "$RECIPE_DIR" && "${CMD[@]}" ) 2>&1 | tee "$LOG"

# ---- point the user at the score files ----
TAG="$(basename "$ASR_CONFIG" .yaml)"
echo "------------------------------------------------------------------------------"
if [[ "$MODE" == "multi" ]]; then
  echo "Scores: $RECIPE_DIR/exp/${TAG}_multilingual_${DURATION}/RESULTS.md"
  echo "        CER (few-shot split): decode_*/*/score_cer/few_shot/{trained,reserved}/result.txt"
else
  # dev/test are ALWAYS the fixed 10min sets (see mono branch). cmn/jpn use word tokens
  # -> score_wer/ (PER); every other language uses char tokens -> score_cer/ (CER).
  SCORE_SUBDIR=score_cer; [[ "$TOKEN_TYPE" == "word" ]] && SCORE_SUBDIR=score_wer
  echo "Scores: $RECIPE_DIR/exp/${TAG}_${LANG_SEL}_${DURATION}/"
  echo "        decode_asr_asr_model_valid.loss.ave.pth/test_10min_${LANG_SEL}/${SCORE_SUBDIR}/result.txt"
fi
echo "RESULT ml_superb1 $TRACK log=$LOG"
