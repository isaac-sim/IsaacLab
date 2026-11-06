# Performance regression comparisons

Compare the PR against its exact base, reuse a verified baseline when safe, and always test the latest PR update fresh.

Open **Checks → Performance regression comparison → Open rendered Performance regression Summary**.

[View the validated CI result](https://github.com/isaac-sim/IsaacLab/actions/runs/41561655180/attempts/1#summary-125679219775).

Example from run **41561655180**, commit **acf2b8cd**:

**🟢 Improved 2 · 🔴 Regressed 2 · ⚪ Not comparable 0**

| Status | Workload | Baseline FPS | PR FPS | Change % |
| --- | --- | ---: | ---: | ---: |
| 🔴 Regressed | cartpole-newton | 1,448,437.85 | 1,429,423.67 | -1.31% |
| 🟢 Improved | shadow-camera-newton-renderer | 36,726.13 | 37,061.07 | +0.91% |
| 🟢 Improved | g1-newton | 43,315.19 | 43,378.02 | +0.15% |
| 🔴 Regressed | shadow-camera-ovphysx-newton | 18,145.77 | 18,039.48 | -0.59% |

Small FPS changes may be normal variation. Both runs use the PR's measuring code.
The separate rolling-history check passed in this run.
