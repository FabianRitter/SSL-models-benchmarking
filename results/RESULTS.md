# Verification status & reference numbers

This table records how far each task's wrapper has been VERIFIED in this repo — it
deliberately does NOT contain benchmark results. *SMOKE* rows were executed end-to-end
on real data with reduced steps (pipeline proof; the metric value is meaningless as a
benchmark and is recorded only in the linked log). *DOCUMENTED* rows are dry-run
verified. To produce real benchmark numbers, run each script with its default settings
(expected runtimes in the docs pages) and compare against the Reference column.

| Benchmark | Task | Metric | Verification status | Reference | Log |
|---|---|---|---|---|---|
| SUPERB | PR (Phoneme Recognition) | PER % ↓ | 8.35 % (**SMOKE** — 200 steps, 10 % of test-clean; not a benchmark) | 3.92 % (WavLM Base+, paper Table I) | logs/superb/pr/smoke_pr_eval.log |
| SUPERB | ASR | WER % ↓ (no LM) | _pending full run_ (smoke OK: `wer=27.66` @1000 steps / 262-utt subset, **SMOKE** — not a result) | 5.59 % (WavLM Base+, no LM — arXiv:2110.13900 Tbl I) | `logs/superb/asr/wavlm_base_plus_asr_full_eval.log` |
| SUPERB | KS | Acc % ↑ | **96.88 (FULL**, default config, 1h30m on 1×H100, 2026-07-19) | 97.37 (WavLM Base+, WavLM paper Tab. I) | logs/superb/ks/wavlm_base_plus_ks_full_eval.log |
| SUPERB | QbE (Query-by-Example) | MTWV ↑ | SMOKE pending (dev, layer 6; PBS job 193842 queued) | 0.0988 (WavLM Base+, paper Table I) | logs/superb/qbe/wavlm_base_plus_qbe_smoke_L6_dev_score.log |
| SUPERB | SID | Acc % ↑ | _pending full run_ (smoke only: acc=0.001697 / 0.17 %, **SMOKE** 100 steps — not a result) | 89.42 (WavLM Base+, WavLM paper Table I) | logs/superb/sid/smoke_sid_eval.log |
| SUPERB | ASV (Speaker Verification) | EER % ↓ | **SMOKE-verified** (300 steps + 2-ckpt test_expdir eval loop, 2026-07-19: best EER 16.07% — pipeline proof, not a result; job 5m34s) | 4.07 % (WavLM Base+, paper Table I) | logs/superb/asv/wavlm_base_plus_asv_eval.log |
| SUPERB | SD (Speaker Diarization) | DER % ↓ | DOCUMENTED (dry-run only; not executed) | 3.50 % (WavLM Base+, paper Table I) | logs/superb/sd/wavlm_base_plus_sd_score.log |
| SUPERB | IC | Acc % ↑ | _pending full run_ (smoke only: acc=0.0635 / 6.35 %, **SMOKE** 300 steps — not a result) | 99.00 (WavLM Base+, WavLM paper Table I) | logs/superb/ic/smoke_ic_eval.log |
| SUPERB | SF (End-to-end Slot Filling) | slot_type_f1 % ↑ | _pending full run_ (smoke OK: 68.68 % @3 000 steps, **SMOKE**) | 90.58 % (WavLM Base+, paper Table I) | logs/superb/sf/wavlm_base_plus_sf_full_eval.log |
| SUPERB | SF (End-to-end Slot Filling) | slot_value_cer % ↓ | _pending full run_ (smoke OK: 55.07 % @3 000 steps, **SMOKE**) | 21.20 % (WavLM Base+, paper Table I) | logs/superb/sf/wavlm_base_plus_sf_full_eval.log |
| SUPERB | ER | Acc % ↑ | _pending full run_ (smoke only: acc_fold1=0.4470 / 44.70 %, **SMOKE** fold1 200 steps — not a result) | 68.65 (WavLM Base+, WavLM paper Table I; 5-fold mean) | logs/superb/er/smoke_er_fold1_eval.log |
| SUPERB-SG | SE (Speech Enhancement) | PESQ ↑ / STOI ↑ / SI-SDRi ↑ | **SMOKE-verified** (200 steps, 2026-07-19: PESQ 2.08 / STOI 0.917 / SI-SDRi 7.69 — pipeline proof, not a result; job 6m47s) | PESQ ~2.64 / STOI ~94.2 (HuBERT-Large ballpark; WavLM leaderboard row not fetched) | logs/superb_sg/se/wavlm_base_plus_se_smoke_eval.log |
| SUPERB-SG | SS (Source Separation) | SI-SDRi dB ↑ | DOCUMENTED (dry-run only; not executed) | ~10.45 dB (HuBERT-Large ballpark; WavLM leaderboard row not fetched) | logs/superb_sg/ss/wavlm_base_plus_ss_eval.log |
| SUPERB-SG | VC a2o (Voice Conversion) | MCD dB ↓ | DOCUMENTED (dry-run only; not executed) | ~7.22 dB (HuBERT-Large ballpark; WavLM leaderboard row not fetched) | logs/superb_sg/vc_a2o/wavlm_base_plus_vc_<spk>_decode.log |
| SUPERB-SG | ST (Speech Translation) | BLEU ↑ | DOCUMENTED (dry-run only; not executed) | ~20.01 (HuBERT-Large ballpark; WavLM leaderboard row not fetched) | logs/superb_sg/st/wavlm_base_plus_st_eval.log |
| ML-SUPERB 1.0 | mono10min xty (full track) | CER ↓ | **62.1 (FULL**, recipe-default schedule, 20 min on 1×H100, 2026-07-19) | no published WavLM row (sanity band only) | logs/ml_superb/mono10min_xty.log |
| ML-SUPERB 1.0 | mono 10min (`--lang xty`) | CER % ↓ | _pending harvest_ — **FULL run of this track** (500×30 iters, not reduced) submitted as PBS **193839** (walltime 06:00) | no WavLM row exists; HuBERT-Base README band eng1 33.8 / deu1 35.1 / jpn 20.6 (10min) | **SUBMITTED / QUEUED** (2026-07-19) · `logs/ml_superb/mlsuperb1_mono10min_xty_smoke.log` |
| ML-SUPERB 1.0 | multi 10min ASR | CER % ↓ (macro, trained split) | _pending_ (~12–24 h/H100) | no WavLM row; band ≈ 20–40 CER (HuBERT-Base class) | **DOCUMENTED** · `logs/ml_superb/mlsuperb1_multi10min.log` |
| ML-SUPERB 1.0 | multi 1h ASR | CER % ↓ (macro, trained split) | _pending_ (~1.5–2.5 days/H100) | no WavLM row; band ≈ HuBERT-Base 1h (eng1 26.7 / deu1 30.2) | **DOCUMENTED** · `logs/ml_superb/mlsuperb1_multi1h.log` |
| ML-SUPERB 1.0 | LID 10min | LID acc % ↑ | _pending_ (~12–24 h/H100) | no WavLM row | **DOCUMENTED** (recipe-native `run_multi.sh … --only_lid true`) |
| ML-SUPERB 2.0 | Standard CER ↓ | _pending harvest_ (PBS 193844) | 24.0 | **SUBMITTED / QUEUED** · `logs/ml_superb/mlsuperb2_baseline_smoke.log` |
| ML-SUPERB 2.0 | Standard LID % ↑ | _pending harvest_ | 74.0 | **SUBMITTED / QUEUED** |
| ML-SUPERB 2.0 | Worst-15 CER ↓ | _pending harvest_ | 71.0 | **SUBMITTED / QUEUED** |
| ML-SUPERB 2.0 | CER StdDev ↓ | _pending harvest_ | 25.5 | **SUBMITTED / QUEUED** |
| ML-SUPERB 2.0 | Dialect CER ↓ | _pending harvest_ | 32.7 | **SUBMITTED / QUEUED** |
| ML-SUPERB 2.0 | Dialect LID % ↑ | _pending harvest_ | 54.0 | **SUBMITTED / QUEUED** |

_Assembled from per-task verification reports; last update 2026-07-19 (ML-SUPERB mono10min xty FULL: CER 62.1)._