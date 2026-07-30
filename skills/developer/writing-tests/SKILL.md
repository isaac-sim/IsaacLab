---
name: isaaclab-writing-tests
description: Writes and prunes Isaac Lab tests against the repository test rules, covering the Kit-less boundary, runtime markers, optional extras, and naming. Use when adding a test, reviewing a test diff, overhauling a package test suite, or deciding whether a test earns its runtime.
audience: developer
status: stable
owners:
  - isaaclab-maintainers
---

# Writing Isaac Lab Tests

## When To Use

Use this skill when adding or changing anything under a `test/` directory, when reviewing a diff
that touches tests, or when overhauling a package's suite.

The governing rules are `## Testing Guidelines` in `AGENTS.md`. This skill is the working
procedure for applying them. Do not use it for `source/isaaclab/test/install_ci/`, which has its
own stricter rules in the same file.

## Workflow

### 1. Decide whether to write a test at all

Search first: `grep -rn "<symbol>" source/<package>/test`.

- An existing test covers the path → extend its assertions.
- An existing test is one input away → add a parametrize case.
- Neither → write a new test.

Reject the test if reverting a production line cannot make it fail. Asserting a default you just
wrote into a config, an attribute you just passed to a constructor, or a statically obvious type
proves nothing.

### 2. Pick the narrowest seam

In order of preference:

1. Plain Python, Torch, Warp, or Newton logic called directly.
2. A configuration object validated without a simulator.
3. A protocol fake from `isaaclab.test.mock_interfaces` or `isaaclab_physx.test.mock_interfaces`.
4. A simulation context.
5. Kit.

Never step down a level for convenience. Needing a fixture that happens to exist in a Kit module
is not a reason to require Kit.

### 3. Place it and mark it

| Module contains | File | Module-level mark |
|---|---|---|
| Direct logic split out of a Kit file | `test_<subject>_unit.py` | none |
| Config contracts only | `test_<subject>_cfg.py` | none |
| Kit imports or `AppLauncher` | `test_<subject>.py` | `pytest.mark.requires_kit` |
| Expensive but Kit-less | any | `pytest.mark.ci_only` |
| Needs an optional extra | any | `pytest.mark.requires_extra("<extra>")` |

Assign `pytestmark` exactly once. A second assignment silently discards the first, so combine
marks into one list. Place it above the `AppLauncher(...)` call, hoisting `import pytest` above
the `AppLauncher` import.

Markers are read by parsing the module, never by importing it, so a function-level `requires_kit`
in a mixed module has no effect on collection.

### 4. Keep the runtime honest

- Behavior tests take one device: `PRIMARY_DEVICE = test_devices()[-1:]`.
- Initialization and transfer tests take paired cases, not a cross product:
  `[(1, test_devices()[0]), (2, test_devices()[-1])]`.
- Independent axes get complementary pairwise cases, not the full matrix.
- If a parameter only flips a cheap attribute, loop over one built scene with `try/finally`
  restore instead of parametrizing.
- Reuse-with-reset beats rebuild, unless leaked state would make assertions order-dependent.

Before narrowing a device axis, check whether the module is marked `device_split` or has an
autouse fixture that skips the device the session is not pinned to. There the device parameter is
how CI shards the file across processes, so collapsing it deletes coverage. Always compare the
passed count before and after; a drop means the change removed tests rather than duplicates.

`ci_only` is for startup cost, large environment matrices, long training, rendering, and expensive
GPU integration. It is not a place to put a slow test you did not want to fix.

### 5. Name it and document it

`test_<subject>_<behavior>_<expected>`, plus a one-line docstring naming the contract. Module names
must be globally unique because test directories are not packages; disambiguate with a package
suffix such as `test_articulation_newton.py`.

### 6. For a regression test, prove it fails first

Revert the production line, run the test and capture the failure, reapply, run again. Report both
directions in the PR description.

## Validation

Run all three lanes for the package you touched:

```bash
uv run --extra test --locked python -m pytest source/<package>/test
uv run --extra test --locked python -m pytest --run-ci-tests --without-kit source/<package>/test
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim --locked --with pytest \
    python -m pytest --run-ci-tests source/<package>/test/<one_file>.py
```

Give the Kit lane a single file. Several modules starting Kit in one pytest process segfaults
with no output at all, which reads like a hang rather than a failure. Never run two `uv run`
commands with different extras concurrently either; they share one `.venv` and will re-sync
packages underneath each other.

A suite that declares an extra needs that extra installed, for example
`uv run --extra test --extra ov --locked python -m pytest source/isaaclab_ovphysx/test`. Extras
that cannot co-resolve must be separate invocations; `--ignore-missing-extras` is only for a
deliberate repository-wide sweep.

Then run the mechanical rule check and the formatter:

```bash
uv run --extra test --locked python -m pytest source/isaaclab/test/test_repo_test_boundary.py
uv run isaaclab -f
```

The second lane must not collect fewer tests than before your change, minus deletions you can
name and justify.

## Maintenance

Keep this skill synchronized with `## Testing Guidelines` in `AGENTS.md`, the enforcement in
`source/isaaclab/test/test_repo_test_boundary.py`, the collection hooks in `conftest.py`, and the
contributor-facing text in `docs/source/refs/contributing.rst`.

The allowlist at `source/isaaclab/test/test_repo_test_boundary_allowlist.txt` may only shrink.
When a package overhaul lands, delete its entries in the same PR.

## References

- [Testing guidelines](../../../AGENTS.md)
- [Contributing guide](../../../docs/source/refs/contributing.rst)
- [Mock interfaces](../../../docs/source/testing/mock_interfaces.rst)
- [Rule enforcement](../../../source/isaaclab/test/test_repo_test_boundary.py)
- [Examples](examples.md)
