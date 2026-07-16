#!/usr/bin/env bash
# =============================================================================
# SUPERB-SG ST — Speech Translation (CoVoST2 En->De) · upstream: WavLM (s3prl)
# Trains a fairseq s2t_transformer decoder on frozen WavLM features and reports
# case-sensitive detokenized BLEU on the test set. Generic local/conda — no
# scheduler assumptions.
#
# REQUIRES fairseq (imported directly by the ST expert; NOT in requirements/all.txt).
# This wrapper gates on `import fairseq` and fails early with an install pointer.
# See docs/superb_sg.md#st for the fairseq install note and the (gated) CoVoST2 /
# Common Voice Corpus 4 data preparation.
#
# Usage:
#   bash scripts/superb_sg/run_wavlm_st.sh [options]
# Options:
#   --data-root PATH   prepared CoVoST En->De data dir (default: data/covost_en_de,
#                      as built by prepare_data/prepare_covo.sh). Overrides
#                      config.downstream_expert.taskrc.data.
#   --upstream NAME    s3prl upstream (default: wavlm_base_plus)
#   --gpu IDS          CUDA_VISIBLE_DEVICES value (default: 0)
#   --exp-name NAME    experiment name (default: <upstream>_st)
#   --stage STAGE      all | prep | train | evaluate (default: all = train+evaluate)
#   --extra-override S extra s3prl -o overrides, appended verbatim after ',,' (default: none)
#   --dry-run          print commands without executing (fairseq gate is skipped)
# Env overrides:
#   S3PRL_ROOT           s3prl checkout (default: <repo>/external/s3prl)
#   SSL_BENCH_CONDA_ENV  conda env name/path (default: ssl-bench-s3prl)
#
# Downstream recipe: -d speech_translation (default config.yaml; arch s2t_transformer).
#   Data-root key: config.downstream_expert.taskrc.data. Checkpoint = dev-best.ckpt.
# Metric: BLEU (↑) — the expert prints "BLEU = <score> ..." and writes
#   result/downstream/<exp>/output-st-test.tsv (also scoreable with count_sacreBLEU.py).
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UPSTREAM="wavlm_base_plus"; GPU="${CUDA_VISIBLE_DEVICES:-0}"; DATA_ROOT="data/covost_en_de"
EXP_NAME=""; STAGE="all"; EXTRA_O=""; DRY_RUN=0

while [[ $# -gt 0 ]]; do case "$1" in
  --data-root) DATA_ROOT="$2"; shift 2;;  --upstream) UPSTREAM="$2"; shift 2;;
  --gpu) GPU="$2"; shift 2;;              --exp-name) EXP_NAME="$2"; shift 2;;
  --stage) STAGE="$2"; shift 2;;          --extra-override) EXTRA_O="$2"; shift 2;;
  --dry-run) DRY_RUN=1; shift;;
  *) echo "unknown option: $1" >&2; exit 2;; esac; done

case "$STAGE" in all|prep|train|evaluate) ;; *) echo "ERROR: --stage must be all|prep|train|evaluate" >&2; exit 2;; esac
EXP_NAME="${EXP_NAME:-${UPSTREAM}_st}"
S3PRL_ROOT="${S3PRL_ROOT:-$REPO_ROOT/external/s3prl}"
[[ -d "$S3PRL_ROOT/s3prl" ]] || { echo "ERROR: s3prl not found at $S3PRL_ROOT (set S3PRL_ROOT or git submodule update --init)" >&2; exit 2; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SSL_BENCH_CONDA_ENV:-ssl-bench-s3prl}"
export CUDA_VISIBLE_DEVICES="$GPU"

# ---- fairseq availability gate (the ST expert does `import fairseq` at module load) ----
# prep needs no fairseq; every executing stage does.
if [[ "$STAGE" == "prep" ]]; then
  :
elif [[ $DRY_RUN -eq 1 ]]; then
  echo "+ python3 -c 'import fairseq'   # fairseq availability gate (skipped in --dry-run)"
else
  python3 -c "import fairseq" 2>/dev/null || {
    echo "ERROR: fairseq is not importable in env '${SSL_BENCH_CONDA_ENV:-ssl-bench-s3prl}'." >&2
    echo "       ST needs fairseq (s2t_transformer). Install it (a pinned build in a dedicated" >&2
    echo "       env is often required on modern torch/CUDA) — see docs/superb_sg.md#st." >&2
    exit 3
  }
fi

LOG_DIR="$REPO_ROOT/logs/superb_sg/st"; mkdir -p "$LOG_DIR"
TRAIN_LOG="$LOG_DIR/${EXP_NAME}_train.log"
EVAL_LOG="$LOG_DIR/${EXP_NAME}_eval.log"
EXPDIR="result/downstream/${EXP_NAME}"
cd "$S3PRL_ROOT/s3prl"
run() { echo "+ $*"; [[ $DRY_RUN -eq 1 ]] || "$@"; }

OVERRIDE="config.downstream_expert.taskrc.data=${DATA_ROOT}"
[[ -n "$EXTRA_O" ]] && OVERRIDE="${OVERRIDE},,${EXTRA_O}"

do_prep() {
  cat <<'PREP'
[ST data prep — run manually; Common Voice Corpus 4 (en) is GATED (Mozilla account)]
  # 1. Obtain Common Voice Corpus 4 (English) from https://commonvoice.mozilla.org/en/datasets
  #    (select the *v4* archive) and extract it; call the extracted dir <CV4_EN_ROOT>
  #    (it must contain en/clips/ and en/validated.tsv).
  # 2. Edit downstream/speech_translation/prepare_data/prepare_covo.sh:
  #      covo_root="<CV4_EN_ROOT>"   (src_lang=en, tgt_lang=de already set)
  # 3. Build tsv + SentencePiece vocab + config into data/covost_en_de:
  cd downstream/speech_translation/prepare_data/ && bash prepare_covo.sh
  # The CoVoST2 en_de translation tsvs are CC0 and fetched by the script; only the
  # Common Voice audio is gated. Result: <S3PRL>/s3prl/data/covost_en_de (default --data-root).
PREP
}

do_train() {
  run bash -c "set -o pipefail; python3 run_downstream.py -m train -u '${UPSTREAM}' -d speech_translation -n '${EXP_NAME}' -o '${OVERRIDE}' 2>&1 | tee '${TRAIN_LOG}'"
}

do_evaluate() {
  local ckpt="${EXPDIR}/dev-best.ckpt"
  [[ $DRY_RUN -eq 1 || -f "$ckpt" ]] || { echo "ERROR: checkpoint not found: $(pwd)/$ckpt (run the train stage first)" >&2; exit 2; }
  run bash -c "set -o pipefail; python3 run_downstream.py -m evaluate -t test -e '${ckpt}' 2>&1 | tee '${EVAL_LOG}'"
  if [[ $DRY_RUN -eq 0 ]]; then
    # The ST expert prints the sacrebleu object: "BLEU = <score> <p1>/<p2>/... (BP ...)".
    BLEU="$(grep -oE 'BLEU = [0-9]+\.[0-9]+' "$EVAL_LOG" | tail -1 | awk '{print $3}')"
    [[ -n "$BLEU" ]] || { echo "ERROR: could not parse 'BLEU =' from $EVAL_LOG (try: python3 downstream/speech_translation/count_sacreBLEU.py --exp-dir $EXPDIR --tsv-file output-st-test.tsv)" >&2; exit 1; }
    echo "RESULT superb_sg st bleu=${BLEU}"
  fi
}

case "$STAGE" in
  prep)     do_prep ;;
  train)    do_train ;;
  evaluate) do_evaluate ;;
  all)      do_train; do_evaluate ;;
esac
