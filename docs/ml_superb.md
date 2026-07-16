# ML-SUPERB — benchmarking guide (ESPnet, `frontend=s3prl`)

**Scope:** ML-SUPERB / ML-SUPERB 2.0 multilingual ASR + language identification (LID),
run through the ESPnet recipe `egs2/ml_superb/asr1` with **WavLM Base+** wired in as
the `frontend=s3prl` feature extractor.

> **Work in progress.** Sections are added here only once verified end-to-end:
> recipe anatomy (data prep → train → decode → score stages) · the exact WavLM
> `frontend=s3prl` config and where it goes (`conf/tuning/`) · dataset acquisition &
> gating · runtime/VRAM on 1×H100 · how to read CER / LID accuracy and where scores
> land · mapping to the paper's numbers · and our executed run (date, exact command,
> metric, **full/reduced** label).

<!-- sections added per task/track as they are verified -->
