# Performance smoke comparisons

Open **Checks → Performance smoke comparison → Open rendered Performance smoke Summary**.

The table shows each workload's baseline FPS, PR FPS, percentage change, and **Improved**, **Regressed**, or **Not comparable** result.
Small FPS changes may be normal variation between runs.

## How it runs

Once CI is activated for a PR, updates rerun automatically and cancel superseded runs.
CI compares GitHub's tested merge revision against its first parent, the base revision.

Both runs use the PR’s code for warmup, timing, and calculating FPS.

The baseline run uses the **base’s task and simulation code**. The PR run uses the **PR’s task and simulation code**.
This compares the two versions using the same measuring method. For example, if the PR changes the timer or FPS calculation, **both runs use that new code**.

We compare:

- Base task/simulation code + new measurement code.
- PR task/simulation code + new measurement code.

We do **not** compare the old measurement code against the new measurement code. Therefore, this comparison cannot tell us whether changing the measurement code itself made the benchmark faster or slower.

Reuse requires verified base results and matching base, workload matrix, launcher settings, and measurement-code digest.
If no reusable results are found, CI measures the base and PR on the same runner. The PR is always measured fresh.

## Comparison safety

Source verification checks workload and measurement code separately and ties both to the results.
Their revisions and the measurement digest appear in the Summary's collapsed source details.

Percentage comparisons require matching task settings, measurement protocols, CPU/GPU models, and device counts.
Missing, incomplete, unverified, or incompatible results remain visible as **Not comparable**.

The existing rolling-history gate and non-PR historical comparison remain separate.
