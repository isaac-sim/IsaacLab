<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Remove the IO Descriptor Export Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the obsolete standalone IO-descriptor export demo while preserving all runtime deprecation warnings and compatibility APIs.

**Architecture:** Delete only the standalone script. Validate the deletion with a file-absence check, scan tracked files for references to the removed path, and run the repository pre-commit suite.

**Tech Stack:** Git, Bash, Isaac Lab pre-commit wrapper

## Global Constraints

- Do not change runtime IO-descriptor implementation or warning behavior.
- Do not add a replacement wrapper or compatibility stub.
- Do not add dependencies.
- Run `./isaaclab.sh -f` before committing.

---

### Task 1: Remove the standalone export demo

**Files:**
- Delete: `scripts/environments/export_IODescriptors.py`

**Interfaces:**
- Consumes: The existing deprecation documentation and runtime `FutureWarning` entry points.
- Produces: A repository with no standalone IO-descriptor export demo.

- [ ] **Step 1: Verify the desired file-absence check fails**

Run:

```bash
test ! -e scripts/environments/export_IODescriptors.py
```

Expected: exit code 1 because the obsolete script still exists.

- [ ] **Step 2: Delete the obsolete script**

Delete `scripts/environments/export_IODescriptors.py` without modifying runtime descriptor code.

- [ ] **Step 3: Verify the file-absence check passes**

Run:

```bash
test ! -e scripts/environments/export_IODescriptors.py
```

Expected: exit code 0.

- [ ] **Step 4: Scan tracked files for stale references**

Run:

```bash
git grep -n "export_IODescriptors.py"
```

Expected: exit code 1 with no matches.

- [ ] **Step 5: Run repository checks**

Run:

```bash
./isaaclab.sh -f
```

Expected: all pre-commit hooks pass.

- [ ] **Step 6: Review and commit the deletion**

Run:

```bash
git diff --check
git diff --stat HEAD^
git add scripts/environments/export_IODescriptors.py
git commit -m "Remove IO descriptor export demo"
```

Expected: the implementation commit deletes only the obsolete script.
