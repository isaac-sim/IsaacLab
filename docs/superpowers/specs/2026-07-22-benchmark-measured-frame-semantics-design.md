<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Benchmark Measured-Frame Semantics

## Context

Runtime and play currently interpret `num_frames` differently from their warm-up
options. Runtime performs warm-up before the measured loop, while play executes
exactly `num_frames` total calls and discards the first `warmup_steps` only from
the environment-step recorder. Consequently, a play request for 250 frames with
one warm-up produces only 249 environment-step timing samples, even though other
throughput inputs still contain all 250 calls.

The CLI should make the requested measured workload independent of warm-up.

## User-facing contract

- `num_frames` is the exact number of measured `env.step()` calls.
- `warmup_frames` and `warmup_steps` are the exact number of preceding calls
  excluded from timing and throughput.
- The total number of calls is `warmup + num_frames`.
- A zero warm-up means the first call is measured.
- Play warm-up calls still advance the environment and policy and contribute to
  reward, episode-length, success, and resource statistics.
- Training remains iteration-controlled. Its warm-up calls continue to consume
  part of the training workload because extending training would change learning.

Examples:

| Workflow | Warm-up | Measured frames | Total calls | Timing samples |
|---|---:|---:|---:|---:|
| Play default | 1 | 250 | 251 | 250 |
| Play without warm-up | 0 | 250 | 250 | 250 |
| Runtime | 50 | 1000 | 1050 | 1000 |
| Runtime without warm-up | 0 | 1000 | 1000 | 1000 |

## Runtime data flow

`run_runtime_warmup()` will execute exactly the requested number of calls instead
of forcing at least one call. `run_runtime_loop()` will continue to execute the
exact measured `num_frames` count.

`StartupTime.first_step` will use the first warm-up sample when warm-up is
nonzero. With zero warm-up it will use the first measured sample. This retains a
first-step diagnostic without silently adding work.

Only the measured loop feeds iteration timing, FPS, environment-step timing, and
throughput aggregation.

## Play data flow

Each play adapter will execute one continuous rollout of
`warmup_steps + num_frames` calls. The existing environment-step recorder will
discard its first `warmup_steps` samples. The shared play loop will continue to
aggregate reward, episode length, and success across the complete rollout so
warm-up does not reset or interrupt episode state.

The raw play-loop step-time series will be split after execution:

- `raw_step_times[0]` supplies `StartupTime.first_step`.
- `raw_step_times[warmup_steps:]` supplies iteration timing and FPS.
- Reward, episode-length, success, and resource statistics retain the complete
  rollout.

This keeps the rollout continuous and avoids a second reset or a new stateful
play-loop abstraction. It also preserves the synchronized recorder's existing
behavior: warm-up calls run under the selected measurement schedule but their
timing and simulation-call accounting are discarded.

## Validation and errors

`num_frames` remains strictly positive and warm-up remains nonnegative. Play no
longer requires `warmup_steps < num_frames`; a warm-up larger than the measured
workload is valid because it adds calls instead of consuming measured calls.

The obsolete parameterized parser test that expects this cross-argument
rejection will be removed. It will be replaced with acceptance and workload
accounting tests rather than leaving the behavior uncovered.

## Documentation and metadata

CLI help and benchmark documentation will state that `num_frames` counts measured
calls and warm-up counts additional excluded calls. Existing `num_frames` and
warm-up metadata fields are sufficient to reconstruct total calls, so no schema
field is added.

## Verification

Regression coverage will verify:

- runtime warm-up executes exactly 0, 1, and 50 calls;
- play accepts `warmup_steps >= num_frames`;
- one warm-up plus 250 measured frames executes 251 calls and records 250 timing
  samples;
- synchronized play with one warm-up plus 10 measured frames executes 11 calls
  and records 10 environment-step samples;
- zero warm-up executes and records exactly `num_frames` calls;
- reward and episode statistics include warm-up activity;
- all four RL play adapters follow the same contract; and
- each regression test fails against the pre-change implementation before the
  implementation is applied, then passes afterward.

The focused benchmark unit and smoke suites and the full pre-commit hooks will be
run before committing implementation changes.
