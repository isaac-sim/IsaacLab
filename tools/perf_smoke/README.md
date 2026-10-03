# Performance smoke comparisons

The **Performance smoke** Summary shows how the PR performed compared to its base.

It's located in **Checks → Performance smoke comparison → Open rendered Performance smoke Summary**

The main table shows:

**🟢 Improved · 🔴 Regressed · ⚪ Not comparable**

with workload, baseline FPS, PR FPS, and percentage change.

Small FPS changes may just be normal variation between runs.

## How it runs

For an activated PR, CI resolves GitHub's exact tested merge revision and uses its first parent as the baseline.

If no valid reusable baseline exists, CI:

1. benchmarks the base,
2. benchmarks the tested PR revision on the same runner,
3. compares all shared compatible workloads.

On later PR updates, the verified baseline can be reused when the base, benchmark matrix, and launcher configuration are unchanged. The current PR revision is always measured fresh. A changed base, matrix, launcher, expired artifact, incomplete result, or failed verification requests a fresh baseline.

Subsequent PR updates rerun automatically after the PR has been activated, and superseded runs are cancelled.

## Comparison safety

Each benchmark runs against the intended checkout, with source verification tied to the produced results.

A workload is only compared when its recorded task/configuration, measurement protocol, and hardware are compatible. Missing, partial, unverified, or incompatible results remain visible as **Not comparable** rather than forcing a percentage delta.

Different CPU/GPU models or device counts are treated as incompatible for percentage comparison.
