#!/usr/bin/env bash
# =============================================================================
# SUPERB-SG VC — Voice Conversion, any-to-one (VCC2020) · upstream: WavLM (s3prl)
# Trains ONE Taco2-AR conversion model per target speaker (TEF1/TEF2/TEM1/TEM2),
# synthesises waveforms with a pretrained neural vocoder, and reports MCD
# (averaged over the target speakers). a2o (any-to-one) is the SUPERB-SG task;
# a2a (any-to-any) is a separate extension — see docs/superb_sg.md#extras.
# Generic local/conda execution — no scheduler assumptions.
#
# Usage:
#   bash scripts/superb_sg/run_wavlm_vc_a2o.sh --data-root PATH [options]
# Options:
#   --data-root PATH   VCC2020 dir (default: downstream/a2o-vc-vcc2020/data/vcc2020,
#                      i.e. where data/data_download.sh puts it). Overrides
#                      config.downstream_expert.datarc.data_root.
#   --trgspk SPEC      TEF1|TEF2|TEM1|TEM2|all (default: all = the 4 task-1 speakers)
#   --vocoder-dir PATH vocoder dir passed to decode.sh
#                      (default: downstream/a2o-vc-vcc2020/pwg_task1)
#   --step N           checkpoint/feature-dump step to decode (default: 10000 = final)
#   --upstream NAME    s3prl upstream (default: wavlm_base_plus)
#   --gpu IDS          CUDA_VISIBLE_DEVICES value (default: 0)
#   --exp-name NAME    experiment-name prefix (default: <upstream>_vc); per-speaker
#                      exp = <exp-name>_<trgspk>
#   --stage STAGE      all | vocoder | train | evaluate (default: all = train+evaluate)
#   --extra-override S extra s3prl -o overrides, appended verbatim after ',,' (default: none)
#   --dry-run          print commands without executing
# Env overrides:
#   S3PRL_ROOT           s3prl checkout (default: <repo>/external/s3prl)
#   SSL_BENCH_CONDA_ENV  conda env name/path (default: ssl-bench-s3prl)
#
# Downstream recipe: -d a2o-vc-vcc2020 (default config.yaml == config_taco2_ar.yaml).
#   Target speaker via config.downstream_expert.trgspk; data via ...datarc.data_root.
# Metric: MCD (↓, averaged over target speakers). decode.sh prints
#   "Mean MCD, f0RMSE, f0CORR, DDUR, CER, WER, accept rate: <mcd> ...". No dev-best
#   checkpoint (save_names=[]); the runner's in-training eval dumps test features to
#   <exp>/<step>/test/hdf5, which decode.sh vocodes and scores. Needs the VC extra
#   deps + a vocoder download.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UPSTREAM="wavlm_base_plus"; GPU="${CUDA_VISIBLE_DEVICES:-0}"
DATA_ROOT="downstream/a2o-vc-vcc2020/data/vcc2020"
EXP_NAME=""; STAGE="all"; EXTRA_O=""; DRY_RUN=0
TRGSPK="all"; VOCODER_DIR="downstream/a2o-vc-vcc2020/pwg_task1"; STEP="10000"

while [[ $# -gt 0 ]]; do case "$1" in
  --data-root) DATA_ROOT="$2"; shift 2;;  --upstream) UPSTREAM="$2"; shift 2;;
  --gpu) GPU="$2"; shift 2;;              --exp-name) EXP_NAME="$2"; shift 2;;
  --stage) STAGE="$2"; shift 2;;          --extra-override) EXTRA_O="$2"; shift 2;;
  --trgspk) TRGSPK="$2"; shift 2;;        --vocoder-dir) VOCODER_DIR="$2"; shift 2;;
  --step) STEP="$2"; shift 2;;            --dry-run) DRY_RUN=1; shift;;
  *) echo "unknown option: $1" >&2; exit 2;; esac; done

case "$STAGE" in all|vocoder|train|evaluate) ;; *) echo "ERROR: --stage must be all|vocoder|train|evaluate" >&2; exit 2;; esac
EXP_NAME="${EXP_NAME:-${UPSTREAM}_vc}"
S3PRL_ROOT="${S3PRL_ROOT:-$REPO_ROOT/external/s3prl}"
[[ -d "$S3PRL_ROOT/s3prl" ]] || { echo "ERROR: s3prl not found at $S3PRL_ROOT (set S3PRL_ROOT or git submodule update --init)" >&2; exit 2; }

# Resolve the target-speaker list.
if [[ "$TRGSPK" == "all" ]]; then
  SPKS=(TEF1 TEF2 TEM1 TEM2)
else
  case "$TRGSPK" in TEF1|TEF2|TEM1|TEM2) SPKS=("$TRGSPK");;
    *) echo "ERROR: --trgspk must be TEF1|TEF2|TEM1|TEM2|all" >&2; exit 2;; esac
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SSL_BENCH_CONDA_ENV:-ssl-bench-s3prl}"
export CUDA_VISIBLE_DEVICES="$GPU"

LOG_DIR="$REPO_ROOT/logs/superb_sg/vc_a2o"; mkdir -p "$LOG_DIR"
cd "$S3PRL_ROOT/s3prl"
run() { echo "+ $*"; [[ $DRY_RUN -eq 1 ]] || "$@"; }

# ---- vocoder: download the pretrained PWG/HiFi-GAN vocoders (gdown-based) ----
# The recipe's downstream/a2o-vc-vcc2020/vocoder_download.sh uses the legacy
# `gdown --id <ID>` form, which gdown>=5 removed (now positional-only) — it fails
# silently and leaves empty vocoder dirs. We fetch the same public artifacts with
# gdown's current syntax so this stage works regardless of the installed gdown
# version. Downloads pwg_task1/, pwg_task2/, hifigan_vctk+vcc2020/ into voc_parent.
do_vocoder() {
  local voc_parent; voc_parent="$(dirname "$VOCODER_DIR")"
  local names=(pwg_task1 pwg_task2 "hifigan_vctk+vcc2020")
  local ids=(11KKux-du6fvsMMB4jNk9YH23YUJjRcDV 1li9DLZGnAheWZrB4oXGo0KWq-fHuFH_l 136tzvhczhHQ4sbaaJUU8UKjkCaca0ub6)
  echo "# downloads ${names[*]} into ${voc_parent} (needs gdown + network)"
  local i
  for i in "${!names[@]}"; do
    local d="${voc_parent}/${names[$i]}"
    run bash -c "set -o pipefail; mkdir -p '${d}'; tmp=\$(mktemp '${d}/XXXXXX.tar.gz'); gdown '${ids[$i]}' -O \"\$tmp\"; tar xzf \"\$tmp\" -C '${d}'; rm -f \"\$tmp\""
  done
}

# ---- train: one model per target speaker ----
do_train() {
  for spk in "${SPKS[@]}"; do
    local exp="${EXP_NAME}_${spk}"
    local o="config.downstream_expert.trgspk=${spk},,config.downstream_expert.datarc.data_root=${DATA_ROOT}"
    [[ -n "$EXTRA_O" ]] && o="${o},,${EXTRA_O}"
    run bash -c "set -o pipefail; python3 run_downstream.py -m train -u '${UPSTREAM}' -d a2o-vc-vcc2020 -n '${exp}' -o '${o}' 2>&1 | tee '${LOG_DIR}/${exp}_train.log'"
  done
}

# ---- evaluate: vocode + score the in-training test-feature dump for --step ----
# The runner's in-training evaluation (its eval_dataloaders include 'test') dumps
# test features to result/downstream/<exp>/<step>/test/hdf5 at every eval_step, so
# a separate `-m evaluate` pass is unnecessary — and on a busy node its fresh WavLM
# re-extraction can stall for hours. We decode that existing dump directly, matching
# the recipe README. --step must therefore be a step at which the in-training eval
# ran (a multiple of config.runner.eval_step; the default 10000 / eval_step 1000 fits).
do_evaluate() {
  local sum=0 n=0
  for spk in "${SPKS[@]}"; do
    local exp="${EXP_NAME}_${spk}" outdir hdf5dir dlog mcd
    outdir="result/downstream/${exp}/${STEP}"
    hdf5dir="${outdir}/test/hdf5"
    dlog="${LOG_DIR}/${exp}_decode.log"
    [[ $DRY_RUN -eq 1 || -d "$hdf5dir" ]] || { echo "ERROR: no test-feature dump at $(pwd)/${hdf5dir}. Run --stage all, or ensure config.runner.eval_step divides --step ${STEP} so the in-training eval produced it." >&2; exit 2; }
    run bash -c "set -o pipefail; ./downstream/a2o-vc-vcc2020/decode.sh '${VOCODER_DIR}' '${outdir}' '${spk}' 2>&1 | tee '${dlog}'"
    if [[ $DRY_RUN -eq 0 ]]; then
      # "Mean MCD, f0RMSE, f0CORR, DDUR, CER, WER, accept rate: <mcd> ..." — MCD is field 1 after ':'.
      mcd="$(grep 'Mean MCD' "$dlog" | tail -1 | sed -E 's/.*rate:[[:space:]]*//' | awk '{print $1}')"
      [[ -n "$mcd" ]] || { echo "ERROR: could not parse MCD for ${spk} from ${dlog}" >&2; exit 1; }
      echo "  ${spk}: MCD=${mcd}"
      sum="$(awk "BEGIN{print $sum + $mcd}")"; n=$((n+1))
    fi
  done
  if [[ $DRY_RUN -eq 0 && $n -gt 0 ]]; then
    local avg; avg="$(awk "BEGIN{printf \"%.2f\", $sum / $n}")"
    echo "RESULT superb_sg vc_a2o mcd=${avg} (mean over ${n} speaker(s))"
  fi
}

case "$STAGE" in
  vocoder)  do_vocoder ;;
  train)    do_train ;;
  evaluate) do_evaluate ;;
  all)      do_train; do_evaluate ;;
esac
