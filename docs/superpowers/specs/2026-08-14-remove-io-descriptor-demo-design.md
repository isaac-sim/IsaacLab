<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Remove the IO Descriptor Export Demo

## Context

IO descriptors are deprecated in Isaac Lab 3.0 and scheduled for removal in
Isaac Lab 3.2. The IO Descriptors 101 tutorial, its navigation entry, and its
generated YAML examples have already been removed by this branch. The remaining
`scripts/environments/export_IODescriptors.py` script still demonstrates and
invokes the deprecated `get_IO_descriptors` API.

## Design

Delete `scripts/environments/export_IODescriptors.py` without adding a wrapper
or compatibility stub. Users of supported RSL-RL/PyTorch deployment workflows
are directed to LEAPP by the existing changelog fragments and deprecation
warnings.

The runtime descriptor APIs and their `FutureWarning` behavior remain
unchanged. This change only removes the obsolete standalone example script.

## Verification

- Search the repository for references to the deleted script and remove any
  live references found.
- Run the repository pre-commit checks required for this PR.
- Review the final diff to confirm that no runtime implementation changed.
