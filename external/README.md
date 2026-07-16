# external/ — pinned upstream submodules

This meta-repo consumes two upstream repositories as **git submodules**, each pinned
to a specific commit so the benchmarks stay reproducible as the upstreams move on.
They are **not wired up yet**: the public forks are created as part of the first
public release, at which point the URLs below are filled in and `git submodule add` is
run against those forks at the pinned SHAs.

| Path | Upstream (fork target) | Pinned commit | Purpose |
|---|---|---|---|
| `external/s3prl` | `s3prl/s3prl` → fork `<your-account>/s3prl` | `ec8064b5889f81ca460fbe2c094ce576a6f120b7` | SUPERB + SUPERB-SG downstreams; provides the WavLM upstream and the `frontend=s3prl` feature extractor. |
| `external/espnet` | `espnet/espnet` → fork `<your-account>/espnet` | `6ed85c0c2be18e2699818b6c042b33ffb7adfa4d` | ML-SUPERB recipe (`egs2/ml_superb/asr1`); consumes the s3prl WavLM upstream as its frontend. |

**Why fork + pin.** Freezing the exact upstream commit these numbers were reproduced
against keeps the workflow reproducible over time. The s3prl fork is expected to stay
byte-identical to upstream (a pure stability pin); the ESPnet fork additionally
receives a WavLM tuning config under `egs2/ml_superb/asr1/conf/tuning/` — a normal,
expected recipe-repo contribution, not a modification of ESPnet internals.

## Wiring the submodules (done at release time)

Once the two forks exist on GitHub:

```bash
git submodule add https://github.com/<your-account>/s3prl.git  external/s3prl
git -C external/s3prl  checkout ec8064b5889f81ca460fbe2c094ce576a6f120b7

git submodule add https://github.com/<your-account>/espnet.git external/espnet
git -C external/espnet checkout 6ed85c0c2be18e2699818b6c042b33ffb7adfa4d

git add .gitmodules external/s3prl external/espnet
git commit -m "chore: pin upstream submodules (s3prl ec8064b, espnet 6ed85c0)"
```

Consumers then clone with `git clone --recursive`, or run
`git submodule update --init --recursive` in an existing checkout.

Until the forks exist, local development uses standalone clones of the two upstreams
checked out at the same pinned SHAs.
