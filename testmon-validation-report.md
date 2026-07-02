<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Testmon affected-test selection validation

Date: 2026-07-02

Pull request: [#6296](https://github.com/isaac-sim/IsaacLab/pull/6296)

Baseline commit: `a1ce9aecf3f57c4988cca146d7bd54c1bf4e6896`

Validation commit: `abd01ffbd3ae44bdb083bfe88eb7f9534dc6ae5a`

## Result

**Affected-test selection and cache persistence work.** The package test proved that a changed
source file selected its dependent tests while unaffected files selected no tests. Installation
CI also used forced Testmon selection and retained the configured smoke tests.

The overall installation workflow was not fully green: ARM passed, while x86 failed the same
conda training smoke test on both attempts because incompatible `pxr` extension wrappers were
loaded. This failure occurred after Testmon correctly selected the smoke test and is not a
selection or cache-restoration failure.

## Method

The PR's cumulative changed-file list contains YAML, so its normal safety logic correctly chooses
a full collection. To exercise selection before merge, two temporary overrides were used:

- Set the package action's default `testmon-mode` to `select`.
- Set both installation jobs' `premerge` input to `true`.

Two behavior-preserving Python edits acted as probes:

- `isaaclab_teleop.control_events.poll_control_events` was rewritten without changing its result.
- The install-CI environment-marker assertion was rewritten equivalently.

The overrides and probes were removed after evidence collection. They are not part of the final
feature diff.

## Evidence

### Package tests

Baseline caches were populated by [Docker + Tests run 28612056512](https://github.com/isaac-sim/IsaacLab/actions/runs/28612056512).
The validation used [Docker + Tests run 28621311359](https://github.com/isaac-sim/IsaacLab/actions/runs/28621311359).

The [isaaclab_teleop validation job](https://github.com/isaac-sim/IsaacLab/actions/runs/28621311359/job/84886407398)
completed successfully. Its per-file summary showed:

- Changed `test_control_events.py`: `4/4` tests selected and passed.
- Each unaffected teleop test file: `0/0` tests selected.
- Failed tests: none.

The cache keys demonstrate restoration from the baseline generation and persistence for the
validation generation:

- Baseline: `testmon-X64-isaac-lab-teleop-test-ea946a165f0ddba67d8217bad1c33a41dd5966d272df2ddca96bd481dee85020-72b694366184acf48bf4f07c7b7b63475af9870f`
- Validation: `testmon-X64-isaac-lab-teleop-test-ea946a165f0ddba67d8217bad1c33a41dd5966d272df2ddca96bd481dee85020-7bca312dd541a68867f24b5dea3f850e89281d92`

This is the expected affected-test behavior: the changed dependency ran its four tests without
running unrelated files.

### Installation tests

Baseline caches were populated by [Installation Tests run 28612057107](https://github.com/isaac-sim/IsaacLab/actions/runs/28612057107).
Validation used [Installation Tests run 28621311311](https://github.com/isaac-sim/IsaacLab/actions/runs/28621311311).

Both architectures invoked pytest with `--testmon --testmon-forceselect`.

| Architecture | Attempt | Result | Pytest summary |
|---|---:|---|---|
| ARM64 | 1 | Pass | `17 passed, 7 skipped, 55 deselected` |
| X64 | 1 | Fail | `1 failed, 16 passed, 7 skipped, 55 deselected` |
| X64 | 2 | Fail | `1 failed, 16 passed, 7 skipped, 55 deselected` |

The repeated x86 failure was:

`test_install_newton_rl_rsl_rl_trains_cartpole` failed during `pxr` initialization with errors
such as `extension class wrapper ... UsdTyped has not been created yet`. The identical selected,
deselected, passed, and skipped counts across both x86 attempts confirm deterministic selection.
The [x86 retry job](https://github.com/isaac-sim/IsaacLab/actions/runs/28621311311/job/84886383176)
contains the reproduced failure.

### Local checks

- Original probe targets: `30 passed`.
- Final package probe targets: `68 passed`.
- Restored production code and all probe targets: `79 passed`.
- Ruff, formatting, YAML, TOML, conflict, license, RST, and other pre-commit hooks passed.
- `git lfs fsck --pointers`: passed. It was run separately because the Windows script launcher
  removed path separators from the hook entry.

## Conclusion

The Testmon feature is validated for:

- cache creation and restoration across PR commits;
- affected package-test selection;
- deselection of unaffected package tests;
- forced premerge selection in installation CI; and
- restoration of smoke tests after Testmon selection.

The remaining x86 conda-training failure is a separate installation/runtime compatibility issue.
If the installation workflow must be entirely green before merge, that smoke test needs to be
fixed or its x86 applicability reconsidered; it should not be hidden by changing Testmon selection.
