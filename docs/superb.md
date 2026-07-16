# SUPERB — task-by-task benchmarking guide (s3prl)

**Scope:** the SUPERB (Interspeech 2021) speech-understanding tasks — PR, KS, IC,
SID, ASV, SD, ER, ASR, QbE, SF — run through s3prl downstreams with **WavLM Base+**
as the upstream feature extractor.

> **Work in progress.** A per-task section is added here only once that task has been
> verified end-to-end. Each section covers: what the task measures · dataset & how to
> get it (incl. gating) · the one-command run · expected runtime/VRAM on 1×H100 · how
> to read the output metric · how it maps to the original paper's number · and our
> executed run (date, exact command, metric, **full/reduced** label).

<!-- sections added per task as they are verified -->
