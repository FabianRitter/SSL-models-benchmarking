#!/usr/bin/env bash
# =============================================================================
# SUPERB ER — Emotion Recognition (IEMOCAP, 5-fold CV) · upstream: WavLM (s3prl)
# Trains and evaluates end-to-end with one command. Emotion Recognition is a
# 4-class utterance-level task (neu / hap+exc / ang / sad) evaluated with
# 5-fold cross-validation; the reported number is the MEAN test accuracy over
# the 5 folds. Generic local/conda execution — no scheduler assumptions.
#
# Usage:
#   bash scripts/superb/run_wavlm_er.sh --data-root PATH [options]
# Options:
#   --data-root PATH   IEMOCAP release root, i.e. IEMOCAP_full_release
#                      (REQUIRED; see docs/superb.md#er--emotion-recognition-iemocap-5-fold)
#   --fold WHICH       fold1|fold2|fold3|fold4|fold5|all   (default: all)
#   --upstream NAME    s3prl upstream (default: wavlm_base_plus)
#   --gpu IDS          CUDA_VISIBLE_DEVICES value (default: 0)
#   --exp-name NAME    experiment-name prefix; per-fold exp = <NAME>_<fold>
#                      (default: <upstream>_er)
#   --stage STAGE      all | train | evaluate (default: all)
#   --extra-override S extra s3prl -o overrides, appended verbatim (default: none)
#   --dry-run          print commands without executing
# Env overrides:
#   S3PRL_ROOT           s3prl checkout (default: <repo>/external/s3prl)
#   SSL_BENCH_CONDA_ENV  conda env name/path (default: ssl-bench-s3prl)
#
# Output — accuracy is a fraction in [0,1]; x100 = acc%. Prints, after the
# evaluated folds:
#   RESULT superb er acc_fold1=<val>      (one line per executed fold)
#   RESULT superb er acc_mean=<val>       (mean over the executed folds)
# Reference (WavLM Base+, WavLM paper Table I): 68.65 acc% = 0.6865 (5-fold mean).
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UPSTREAM="wavlm_base_plus"; GPU="${CUDA_VISIBLE_DEVICES:-0}"; DATA_ROOT=""
EXP_NAME=""; STAGE="all"; EXTRA_O=""; DRY_RUN=0; FOLD="all"
while [[ $# -gt 0 ]]; do case "$1" in
  --data-root) DATA_ROOT="$2"; shift 2;;    --fold) FOLD="$2"; shift 2;;
  --upstream) UPSTREAM="$2"; shift 2;;      --gpu) GPU="$2"; shift 2;;
  --exp-name) EXP_NAME="$2"; shift 2;;      --stage) STAGE="$2"; shift 2;;
  --extra-override) EXTRA_O="$2"; shift 2;; --dry-run) DRY_RUN=1; shift;;
  *) echo "unknown option: $1" >&2; exit 2;; esac; done
[[ -n "$DATA_ROOT" ]] || { echo "ERROR: --data-root is required (see docs/superb.md#er--emotion-recognition-iemocap-5-fold)" >&2; exit 2; }
case "$STAGE" in all|train|evaluate) ;; *) echo "ERROR: --stage must be all|train|evaluate" >&2; exit 2;; esac
case "$FOLD" in
  all) FOLDS=(fold1 fold2 fold3 fold4 fold5);;
  fold1|fold2|fold3|fold4|fold5) FOLDS=("$FOLD");;
  *) echo "ERROR: --fold must be one of fold1..fold5 or all" >&2; exit 2;;
esac
EXP_NAME="${EXP_NAME:-${UPSTREAM}_er}"
S3PRL_ROOT="${S3PRL_ROOT:-$REPO_ROOT/external/s3prl}"
[[ -d "$S3PRL_ROOT/s3prl" ]] || { echo "ERROR: s3prl not found at $S3PRL_ROOT (set S3PRL_ROOT or git submodule update --init)" >&2; exit 2; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SSL_BENCH_CONDA_ENV:-ssl-bench-s3prl}"
export CUDA_VISIBLE_DEVICES="$GPU"
LOG_DIR="$REPO_ROOT/logs/superb/er"; mkdir -p "$LOG_DIR"
cd "$S3PRL_ROOT/s3prl"

# run_tee <logfile> <cmd...> : echo, run, tee combined stdout+stderr to logfile,
# and return the *command's* exit status (not tee's), honoring --dry-run.
run_tee() {
  local log="$1"; shift
  echo "+ $* 2>&1 | tee $log"
  if [[ $DRY_RUN -eq 1 ]]; then return 0; fi
  local rc=0
  set +e
  "$@" 2>&1 | tee "$log"
  rc="${PIPESTATUS[0]}"
  set -e
  return "$rc"
}

declare -a DONE_FOLDS DONE_ACCS
for fold in "${FOLDS[@]}"; do
  EXP="${EXP_NAME}_${fold}"
  # Data root + test fold. Single quotes around the fold value make s3prl's
  # override() eval("'fold1'") -> the clean string fold1 (helper.py:84).
  OVR="config.downstream_expert.datarc.root=${DATA_ROOT},,config.downstream_expert.datarc.test_fold='${fold}'"
  if [[ -n "$EXTRA_O" ]]; then OVR="${OVR},,${EXTRA_O}"; fi
  TRAIN_LOG="$LOG_DIR/${EXP}_train.log"
  EVAL_LOG="$LOG_DIR/${EXP}_eval.log"
  CKPT="result/downstream/${EXP}/dev-best.ckpt"

  if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
    echo "=== [ER] TRAIN ${fold} (exp=${EXP}) ==="
    run_tee "$TRAIN_LOG" python3 run_downstream.py -m train -u "$UPSTREAM" -d emotion -n "$EXP" -o "$OVR"
  fi

  if [[ "$STAGE" == "all" || "$STAGE" == "evaluate" ]]; then
    echo "=== [ER] EVALUATE ${fold} (ckpt=${CKPT}) ==="
    if [[ $DRY_RUN -eq 0 && ! -f "$CKPT" ]]; then
      echo "ERROR: checkpoint not found: $CKPT (run the train stage first)" >&2; exit 1
    fi
    # evaluate restores -d/-u/root/test_fold from the ckpt; -t defaults to 'test'
    # so emotion/expert.py prints 'test acc: <fraction>'.
    run_tee "$EVAL_LOG" python3 run_downstream.py -m evaluate -e "$CKPT"
    if [[ $DRY_RUN -eq 0 ]]; then
      acc="$(grep -oE 'test acc: [0-9]+\.?[0-9]*' "$EVAL_LOG" | tail -1 | awk '{print $3}')" || true
      [[ -n "$acc" ]] || { echo "ERROR: could not parse 'test acc:' from $EVAL_LOG" >&2; exit 1; }
      DONE_FOLDS+=("$fold"); DONE_ACCS+=("$acc")
      echo "--- [ER] ${fold} test acc = ${acc}"
    fi
  fi
done

# ---- aggregate + print RESULT lines (skip for train-only / dry-run) ----
if [[ $DRY_RUN -eq 0 && "$STAGE" != "train" && ${#DONE_ACCS[@]} -gt 0 ]]; then
  echo ""
  echo "================= ER RESULTS (${UPSTREAM}) ================="
  for i in "${!DONE_FOLDS[@]}"; do
    echo "RESULT superb er acc_${DONE_FOLDS[$i]}=${DONE_ACCS[$i]}"
  done
  mean="$(printf '%s\n' "${DONE_ACCS[@]}" | awk '{s+=$1; n++} END{if(n>0) printf "%.6f", s/n}')"
  echo "RESULT superb er acc_mean=${mean}"
  pct="$(awk -v m="$mean" 'BEGIN{printf "%.2f", m*100}')"
  echo "# ER mean test acc over ${#DONE_ACCS[@]} fold(s) = ${mean} (${pct} acc%). Reference WavLM Base+ = 68.65 acc% (5-fold mean)."
fi
