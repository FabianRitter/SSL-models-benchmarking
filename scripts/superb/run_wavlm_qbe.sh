#!/usr/bin/env bash
# =============================================================================
# SUPERB QbE — Query-by-Example Spoken Term Detection · upstream: WavLM (s3prl)
# NO training. Extracts frozen features and runs DTW per hidden layer over
# QUESST14, then scores with the dataset's shipped NIST scorer (MTWV).
# SUPERB protocol: sweep every hidden layer on the dev set, pick the best layer,
# report that layer's test-set MTWV. Generic local/conda execution — no
# scheduler assumptions.
#
# Usage:
#   bash scripts/superb/run_wavlm_qbe.sh --data-root PATH [options]
# Options:
#   --data-root PATH   QUESST14 root: the extracted 'quesst14Database' dir. Must
#                      contain scoring/score-TWV-Cnxe.sh plus the shipped
#                      groundtruth_quesst14_{dev,eval} and language_key lists.
#                      REQUIRED. See docs/superb.md#qbe.
#   --upstream NAME    s3prl upstream (default: wavlm_base_plus)
#   --layer SPEC       'all' (sweep 0..num-layers-1) or a single integer layer
#                      index (indexed from 0; -1 = last layer) (default: all)
#   --num-layers N     number of hidden states swept by 'all'
#                      (default: 13 = WavLM Base/Base+; WavLM Large = 25)
#   --dist-fn NAME     DTW distance: cosine|cosine_exp|cityblock|euclidean
#                      (default: cosine; the config default is cosine_exp)
#   --split WHICH      dev | test | both (default: both). 'dev' evaluates+scores
#                      only the dev queries (useful for a single-layer smoke);
#                      'both' is the benchmark protocol (best layer picked on dev,
#                      that layer's test MTWV reported).
#   --gpu IDS          CUDA_VISIBLE_DEVICES (default: 0). GPU is used only for
#                      feature extraction; the DTW sweep is CPU-bound.
#   --exp-name NAME    experiment-name prefix (default: <upstream>_qbe)
#   --stage STAGE      all | dtw | score (default: all)
#   --extra-override S extra s3prl -o overrides, appended verbatim after ',,' (default: none)
#   --dry-run          print the full command plan without executing
# Env overrides:
#   S3PRL_ROOT           s3prl checkout (default: <repo>/external/s3prl)
#   SSL_BENCH_CONDA_ENV  conda env name/path (default: ssl-bench-s3prl)
#
# Downstream recipe: -d quesst14_dtw (no -c; no training, no checkpoint).
#   Data-root override key: config.downstream_expert.datarc.dataset_root
#   DTW distance override key: config.downstream_expert.dtwrc.dist_method
# Metric: MTWV (maximum term weighted value) from score-TWV-Cnxe.sh; higher is
#   better. Each DTW run writes result/downstream/<exp>/benchmark.stdlist.xml,
#   which the scorer turns into TWV/MTWV/Cnxe.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UPSTREAM="wavlm_base_plus"; GPU="${CUDA_VISIBLE_DEVICES:-0}"; DATA_ROOT=""
EXP_NAME=""; STAGE="all"; EXTRA_O=""; DRY_RUN=0
LAYER_SPEC="all"; NUM_LAYERS=13; DIST_FN="cosine"; SPLIT="both"

while [[ $# -gt 0 ]]; do case "$1" in
  --data-root) DATA_ROOT="$2"; shift 2;;  --upstream) UPSTREAM="$2"; shift 2;;
  --gpu) GPU="$2"; shift 2;;              --exp-name) EXP_NAME="$2"; shift 2;;
  --stage) STAGE="$2"; shift 2;;          --extra-override) EXTRA_O="$2"; shift 2;;
  --layer) LAYER_SPEC="$2"; shift 2;;     --num-layers) NUM_LAYERS="$2"; shift 2;;
  --dist-fn) DIST_FN="$2"; shift 2;;      --split) SPLIT="$2"; shift 2;;
  --dry-run) DRY_RUN=1; shift;;
  *) echo "unknown option: $1" >&2; exit 2;; esac; done

[[ -n "$DATA_ROOT" ]] || { echo "ERROR: --data-root is required (QUESST14 root; see docs/superb.md#qbe)" >&2; exit 2; }
case "$STAGE" in all|dtw|score) ;; *) echo "ERROR: --stage must be all|dtw|score" >&2; exit 2;; esac
case "$SPLIT" in dev|test|both) ;; *) echo "ERROR: --split must be dev|test|both" >&2; exit 2;; esac
EXP_NAME="${EXP_NAME:-${UPSTREAM}_qbe}"
S3PRL_ROOT="${S3PRL_ROOT:-$REPO_ROOT/external/s3prl}"
[[ -d "$S3PRL_ROOT/s3prl" ]] || { echo "ERROR: s3prl not found at $S3PRL_ROOT (set S3PRL_ROOT or git submodule update --init)" >&2; exit 2; }
S3PRL_RUN="$S3PRL_ROOT/s3prl"

# Build the list of layers to evaluate.
if [[ "$LAYER_SPEC" == "all" ]]; then
  LAYERS=(); for ((l=0; l<NUM_LAYERS; l++)); do LAYERS+=("$l"); done
else
  [[ "$LAYER_SPEC" =~ ^-?[0-9]+$ ]] || { echo "ERROR: --layer must be 'all' or an integer" >&2; exit 2; }
  LAYERS=("$LAYER_SPEC")
fi

# Build the list of splits to evaluate (dev uses the dev query list + the
# groundtruth_quesst14_dev references; test uses the eval queries + eval refs).
case "$SPLIT" in
  both) SPLITS=(dev test);;
  dev)  SPLITS=(dev);;
  test) SPLITS=(test);;
esac

# The scorer ships inside the dataset; sanity-check it (skipped in --dry-run).
if [[ $DRY_RUN -eq 0 && ( "$STAGE" == "all" || "$STAGE" == "score" ) ]]; then
  [[ -x "$DATA_ROOT/scoring/score-TWV-Cnxe.sh" ]] || { echo "ERROR: scorer not found/executable: $DATA_ROOT/scoring/score-TWV-Cnxe.sh (is --data-root the extracted quesst14Database?)" >&2; exit 2; }
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SSL_BENCH_CONDA_ENV:-ssl-bench-s3prl}"
export CUDA_VISIBLE_DEVICES="$GPU"

LOG_DIR="$REPO_ROOT/logs/superb/qbe"; mkdir -p "$LOG_DIR"
cd "$S3PRL_RUN"
run() { echo "+ $*"; [[ $DRY_RUN -eq 1 ]] || "$@"; }

# DTW overrides: dataset_root + distance method, plus any extra appended verbatim.
DTW_O="config.downstream_expert.datarc.dataset_root=${DATA_ROOT},,config.downstream_expert.dtwrc.dist_method=${DIST_FN}"
[[ -n "$EXTRA_O" ]] && DTW_O="${DTW_O},,${EXTRA_O}"

# gt_for SPLIT — map a split name to its shipped groundtruth reference dir.
gt_for() { [[ "$1" == "dev" ]] && echo "groundtruth_quesst14_dev" || echo "groundtruth_quesst14_eval"; }

# ---- DTW stage: for every layer, extract features + DTW on each split ----
do_dtw() {
  for L in "${LAYERS[@]}"; do
    for SP in "${SPLITS[@]}"; do
      run bash -c "set -o pipefail; python3 run_downstream.py -m evaluate -t ${SP} -u '${UPSTREAM}' -l ${L} -d quesst14_dtw -n '${EXP_NAME}_L${L}_${SP}' -o '${DTW_O}' 2>&1 | tee '${LOG_DIR}/${EXP_NAME}_L${L}_${SP}.log'"
    done
  done
}

# grep_mtwv FILE — pull the MTWV float out of a scorer output log. The shipped
# QUESST14 NIST scorer (MediaEvalQUESST2014.jar) prints a line of the form
#   "actTWV: <v>  maxTWV: <v>  Threshold: <t>"
# where maxTWV IS the MTWV (maximum term weighted value). Extract the float that
# immediately follows "maxTWV:"; fall back to a generic maximum-TWV / MTWV line
# for other scorer builds. Always exits 0 (empty on no match) so it is safe under
# `set -euo pipefail`.
grep_mtwv() {
  local v
  v="$(grep -oiE 'maxTWV:[[:space:]]*[-0-9]+\.[0-9]+' "$1" 2>/dev/null | grep -oE '[-0-9]+\.[0-9]+' | tail -n1)"
  [[ -n "$v" ]] || v="$(grep -iE 'maximum.*twv|mtwv' "$1" 2>/dev/null | grep -oE '[-0-9]+\.[0-9]+' | tail -n1)"
  printf '%s' "$v"
}

# ---- Score stage: run the dataset's NIST scorer per layer, per split ----
do_score() {
  declare -A DEV_MTWV TEST_MTWV
  for L in "${LAYERS[@]}"; do
    for SP in "${SPLITS[@]}"; do
      local exp slog gt m
      exp="$S3PRL_RUN/result/downstream/${EXP_NAME}_L${L}_${SP}"
      slog="${LOG_DIR}/${EXP_NAME}_L${L}_${SP}_score.log"
      gt="$(gt_for "$SP")"
      run bash -c "cd '${DATA_ROOT}/scoring' && ./score-TWV-Cnxe.sh '${exp}' ${gt} -10 2>&1 | tee '${slog}'"
      [[ $DRY_RUN -eq 1 ]] && continue
      m="$(grep_mtwv "$slog")"
      echo "layer ${L} ${SP}: MTWV=${m:-?}"
      if [[ "$SP" == "dev" ]]; then DEV_MTWV[$L]="$m"; else TEST_MTWV[$L]="$m"; fi
    done
  done
  [[ $DRY_RUN -eq 1 ]] && return 0

  # SUPERB protocol: pick the layer that maximises dev MTWV, report its test MTWV.
  local best_layer="" best_dev=""
  for L in "${LAYERS[@]}"; do
    local d="${DEV_MTWV[$L]:-}"
    [[ -n "$d" ]] || continue
    if [[ -z "$best_dev" ]] || awk "BEGIN{exit !($d > $best_dev)}"; then
      best_dev="$d"; best_layer="$L"
    fi
  done
  if [[ -n "$best_layer" ]]; then
    local bt="${TEST_MTWV[$best_layer]:-}"
    echo "Best dev layer = ${best_layer} (dev MTWV=${best_dev})."
    if [[ -n "$bt" ]]; then
      echo "RESULT superb qbe mtwv=${bt} (test; best-dev layer=${best_layer})"
    else
      echo "RESULT superb qbe mtwv=${best_dev} (dev; best-dev layer=${best_layer}; test not scored)"
    fi
    return 0
  fi

  # No dev scores (e.g. --split test): cannot apply best-dev selection.
  local best_test="" best_tlayer=""
  for L in "${LAYERS[@]}"; do
    local t="${TEST_MTWV[$L]:-}"
    [[ -n "$t" ]] || continue
    if [[ -z "$best_test" ]] || awk "BEGIN{exit !($t > $best_test)}"; then
      best_test="$t"; best_tlayer="$L"
    fi
  done
  if [[ -n "$best_tlayer" ]]; then
    echo "WARNING: dev not scored — cannot apply the best-dev protocol; reporting best test layer only." >&2
    echo "RESULT superb qbe mtwv=${best_test} (test; layer=${best_tlayer}; no best-dev selection)"
  else
    echo "WARNING: could not parse MTWV from scorer logs in ${LOG_DIR} — inspect *_score.log manually (the MTWV line format may differ in your scorer build)." >&2
    echo "RESULT superb qbe mtwv=NA"
  fi
}

case "$STAGE" in
  dtw)   do_dtw ;;
  score) do_score ;;
  all)   do_dtw; do_score ;;
esac
