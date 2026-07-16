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
LAYER_SPEC="all"; NUM_LAYERS=13; DIST_FN="cosine"

while [[ $# -gt 0 ]]; do case "$1" in
  --data-root) DATA_ROOT="$2"; shift 2;;  --upstream) UPSTREAM="$2"; shift 2;;
  --gpu) GPU="$2"; shift 2;;              --exp-name) EXP_NAME="$2"; shift 2;;
  --stage) STAGE="$2"; shift 2;;          --extra-override) EXTRA_O="$2"; shift 2;;
  --layer) LAYER_SPEC="$2"; shift 2;;     --num-layers) NUM_LAYERS="$2"; shift 2;;
  --dist-fn) DIST_FN="$2"; shift 2;;      --dry-run) DRY_RUN=1; shift;;
  *) echo "unknown option: $1" >&2; exit 2;; esac; done

[[ -n "$DATA_ROOT" ]] || { echo "ERROR: --data-root is required (QUESST14 root; see docs/superb.md#qbe)" >&2; exit 2; }
case "$STAGE" in all|dtw|score) ;; *) echo "ERROR: --stage must be all|dtw|score" >&2; exit 2;; esac
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

# ---- DTW stage: for every layer, extract features + DTW on dev and test ----
do_dtw() {
  for L in "${LAYERS[@]}"; do
    run bash -c "set -o pipefail; python3 run_downstream.py -m evaluate -t dev  -u '${UPSTREAM}' -l ${L} -d quesst14_dtw -n '${EXP_NAME}_L${L}_dev'  -o '${DTW_O}' 2>&1 | tee '${LOG_DIR}/${EXP_NAME}_L${L}_dev.log'"
    run bash -c "set -o pipefail; python3 run_downstream.py -m evaluate -t test -u '${UPSTREAM}' -l ${L} -d quesst14_dtw -n '${EXP_NAME}_L${L}_test' -o '${DTW_O}' 2>&1 | tee '${LOG_DIR}/${EXP_NAME}_L${L}_test.log'"
  done
}

# grep_mtwv FILE — pull an MTWV float out of a scorer output log (best-effort;
# the shipped NIST scorer prints a line naming the maximum TWV).
grep_mtwv() {
  grep -iE 'maximum.*twv|mtwv' "$1" 2>/dev/null | grep -oE '[-0-9]+\.[0-9]+' | tail -n1
}

# ---- Score stage: run the dataset's NIST scorer per layer, per split ----
do_score() {
  local best_layer="" best_dev="" best_test=""
  for L in "${LAYERS[@]}"; do
    local dev_exp test_exp dev_slog test_slog
    dev_exp="$S3PRL_RUN/result/downstream/${EXP_NAME}_L${L}_dev"
    test_exp="$S3PRL_RUN/result/downstream/${EXP_NAME}_L${L}_test"
    dev_slog="${LOG_DIR}/${EXP_NAME}_L${L}_dev_score.log"
    test_slog="${LOG_DIR}/${EXP_NAME}_L${L}_test_score.log"
    run bash -c "cd '${DATA_ROOT}/scoring' && ./score-TWV-Cnxe.sh '${dev_exp}'  groundtruth_quesst14_dev  -10 2>&1 | tee '${dev_slog}'"
    run bash -c "cd '${DATA_ROOT}/scoring' && ./score-TWV-Cnxe.sh '${test_exp}' groundtruth_quesst14_eval -10 2>&1 | tee '${test_slog}'"
    [[ $DRY_RUN -eq 1 ]] && continue
    local dev_mtwv test_mtwv
    dev_mtwv="$(grep_mtwv "$dev_slog")"; test_mtwv="$(grep_mtwv "$test_slog")"
    echo "layer ${L}: dev MTWV=${dev_mtwv:-?}  test MTWV=${test_mtwv:-?}"
    if [[ -n "$dev_mtwv" ]]; then
      if [[ -z "$best_dev" ]] || awk "BEGIN{exit !($dev_mtwv > $best_dev)}"; then
        best_dev="$dev_mtwv"; best_test="$test_mtwv"; best_layer="$L"
      fi
    fi
  done
  if [[ $DRY_RUN -eq 0 ]]; then
    if [[ -n "$best_layer" ]]; then
      echo "Best dev layer = ${best_layer} (dev MTWV=${best_dev}); report its test MTWV."
      echo "RESULT superb qbe mtwv=${best_test:-NA} (best-dev layer=${best_layer})"
    else
      echo "WARNING: could not parse MTWV from scorer logs in ${LOG_DIR} — inspect *_score.log manually (the MTWV line format may differ in your scorer build)." >&2
      echo "RESULT superb qbe mtwv=NA"
    fi
  fi
}

case "$STAGE" in
  dtw)   do_dtw ;;
  score) do_score ;;
  all)   do_dtw; do_score ;;
esac
