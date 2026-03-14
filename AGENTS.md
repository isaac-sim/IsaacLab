# IsaacLab Guidelines

## Breaking API changes

- **Breaking changes require a deprecation first.** Do not remove or rename public API symbols without deprecating them in a prior release.

## API design rules (naming + structure)

- **Group by common prefix for discoverability (autocomplete).**
  - **Classes**: group by domain concept — `ActuatorNetLSTM`, `ActuatorNetMLP` (not `LSTMActuatorNet`, `MLPActuatorNet`).
  - **Methods**: group by noun before modifier — `set_joint_position_target()` (not `set_target_joint_position()`).
- **Method names are `snake_case`.**
- **CLI arguments are `snake_case`.**
- **Prefer nested classes when self-contained.**
  - If a helper type or an enum is only meaningful inside one parent class and doesn't need a public identity, define it as a nested class instead of creating a new top-level class/module.
- **Follow PEP 8 for Python code.**
- **Use modern Python type-hint syntax.**
  - Prefer PEP 604 unions: `x | y`, `x | None`. Do not use `typing.Union` or `typing.Optional`.
- **Use specific type hints for public interfaces.**
  - For torch tensors, annotate with `torch.Tensor`. For Warp arrays, annotate concrete dtypes (e.g., `wp.array(dtype=wp.vec3)`) rather than generic `object`.
  - Prefer consistent parameter names across base/override APIs (e.g., `xforms`, `scales`, `colors`, `materials`).
- **Use Google-style docstrings.**
  - Write clear, concise docstrings that explain what the function does, its parameters, and its return value.
  - Keep argument/return types in function annotations, not inline in docstrings.
  - In `Args:` entries, use `name: description` (not `name (Type): description`).
  - Use Sphinx cross-reference roles for symbol references (e.g. `:class:`, `:meth:`, `:attr:`, `:paramref:`), but keep targets as short as possible.
  - Within the same class/module, prefer short local references (e.g. `:meth:\`set_joint_position_target\``, `:attr:\`num_joints\``) over fully qualified paths.
  - If qualification is needed, prefer public API paths (e.g. `isaaclab.assets.Articulation`) and do not use internal `_src` or private module paths in Sphinx role targets.
- **State SI units for all physical quantities in docstrings.**
  - Use inline `[unit]` notation, e.g. `"""Particle positions [m], shape [particle_count, 3], float."""`.
  - For joint-type-dependent quantities use `[m or rad, depending on joint type]`.
  - For spatial vectors annotate both components, e.g. `[N, N·m]`.
  - For compound arrays list per-component units, e.g. `[0] k_mu [Pa], [1] k_lambda [Pa], ...`.
  - When a parameter's interpretation varies across solvers, document each solver's convention instead of a single unit.
  - Skip non-physical fields (indices, keys, counts, flags).
  - This rule applies to **public API docstrings only**, not test docstrings.
- **Keep the documentation up-to-date.**
  - When adding new files or symbols that are part of the public-facing API, make sure to keep the auto-generated documentation updated by running `./isaaclab.sh -d`.

## Dependencies

- **Avoid adding new required dependencies.** IsaacLab's core should remain lightweight and minimize external requirements.
- **Strongly prefer not adding new optional dependencies.** If additional functionality requires a new package, carefully consider whether the benefit justifies the added complexity and maintenance burden. When possible, implement functionality using existing dependencies, including Warp functions and kernels, NumPy, or the standard library.

## Tooling: prefer `./isaaclab.sh -p` for running, testing, and benchmarking

We use a wrapped python call within `./isaaclab.sh`.

- **Use `./isaaclab.sh -p -c` for inline Python**: When running one-off Python commands, use `./isaaclab.sh -p -c "..."` instead of `python3 -c "..."`.
- **Use `./isaaclab.sh -p`** to run standalone Python scripts without a `pyproject.toml` (e.g., in CI after switching to a branch with no project files).

### Run tests

```bash
# run all tests (extremely heavy, should be avoided).
./isaaclab.sh -t

# run a specific test file by name
./isaaclab.sh -p -m pytest PATH_TO_TEST

# run a specific example test
./isaaclab.sh -p -m pytest PATH_TO_TEST::METHOD
```

### Pre-commit (lint/format hooks)

**CRITICAL: Always run pre-commit hooks BEFORE committing, not after.**

Proper workflow:
1. Make your code changes
2. Run `./isaaclab.sh -f` to check ALL files
3. If pre-commit modifies any files (e.g., formatting), review the changes
4. Stage the modified files with `git add`
5. Run `./isaaclab.sh -f` again to ensure all checks pass
6. Only then create your commit with `git commit`

```bash
# Run pre-commit checks on all files
./isaaclab.sh -f
```

**Common mistake to avoid:**
- Don't commit first and then run pre-commit (requires amending commits)
- Do run pre-commit before committing (clean workflow)

**When reviewing code** (e.g. via a code-reviewer agent), always run `./isaaclab.sh -f` as part of the review to catch formatting or lint issues early.

## Changelog

- **Update `CHANGELOG.rst` for every change** targeting the source directory. Each extension has its own changelog at `source/<package>/docs/CHANGELOG.rst` (e.g. `source/isaaclab/docs/CHANGELOG.rst`, `source/isaaclab_physx/docs/CHANGELOG.rst`).
- **Always create a new version heading.** Never add entries to an existing version — they are released and immutable. Bump the patch version (e.g. `1.5.0` → `1.5.1`) and use today's date.
- **Bump `config/extension.toml` to match.** When creating a new changelog version, update the `version` field in `source/<package>/config/extension.toml` to the same version string.
- **Determine which changelog(s) to update** by looking at which `source/<package>/` directories your changes touch. A single PR may require entries in multiple changelogs.
- Use **past tense** matching the section header: "Added X", "Fixed Y", "Changed Z".
- Place entries under the correct category: `Added`, `Changed`, `Deprecated`, `Removed`, or `Fixed`.
- Avoid internal implementation details users wouldn't understand.
- **For `Deprecated`, `Changed`, and `Removed` entries, include migration guidance.**
  - Example: "Deprecated `Articulation.A` in favor of `Articulation.B`."
- Use Sphinx cross-reference roles for class/method/module names.

### RST formatting reference

```
X.Y.Z (YYYY-MM-DD)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~package.ClassName` to support feature X.

Fixed
^^^^^

* Fixed edge case in :meth:`~package.ClassName.method` where input was
  not validated, causing ``AttributeError`` at runtime.
```

Key formatting rules:
- Version heading: underline with `~` (tildes), must be at least as long as the heading text.
- Category heading: underline with `^` (carets).
- Entries: `* ` prefix, continuation lines indented by 2 spaces.
- Blank line between the last entry and the next version heading.

## Commit and Pull Request Guidelines

Follow conventional commit message practices.

- **Use feature branches**: All development work should be on branches named `<username>/feature-desc` (e.g., `jdoe/docs-versioning`). Do not commit directly to `main`.
- Keep commits focused and atomic—one logical change per commit.
- Reference related issues in commit messages when applicable.
- **When iterating on PR feedback**, prefer adding new commits over amending existing ones. This avoids force-pushing and lets the reviewer easily verify each change request was addressed.
- **Do not include AI attribution or co-authorship lines** (e.g., "Co-Authored-By: Claude...") in commit messages. Commits should represent human contributions without explicit AI attribution.
- **Commit message format**:
  - Separate subject from body with a blank line
  - Subject: imperative mood, capitalized, ~50 chars, no trailing period
    - Write as a command: "Fix bug" not "Fixed bug" or "Fixes bug"
    - Test: "If applied, this commit will _[your subject]_"
  - Body: wrap at 72 chars, explain _what_ and _why_ (not _how_—the diff shows that)

## File headers and copyright

- New files must use the current year (2026) in the SPDX copyright header:
  ```
  # Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
  # All rights reserved.
  #
  # SPDX-License-Identifier: BSD-3-Clause
  ```
- Do not change the year in existing file headers.

## Issue Triage Workflow

Automated pipeline for triaging, reproducing, and fixing GitHub bug reports. Trigger it for a single issue with:

```bash
claude "Run the issue triage workflow for issue #<NUMBER>"
```

The workflow chains three skills in `.agent/skills/` and follows this decision tree:

```
Fetch issue #N
├─ Not a bug report → Skip
├─ Missing reproduction steps (and no other useful details)
│  └─ Comment asking for steps → STOP
├─ Has reproduction steps (use latest develop if commit is missing)
│  └─ Checkout commit, run repro steps
│     ├─ Cannot reproduce
│     │  └─ Comment with what was tried, ask for details → STOP
│     └─ Reproduces
│        └─ Checkout latest develop, re-run
│           ├─ Fixed on latest
│           │  └─ Comment with evidence, close issue → STOP
│           └─ Still broken
│              └─ Create branch, fix, test, changelog, pre-commit, PR → DONE
```

### Skills

| Skill | Path | Purpose |
|-------|------|---------|
| `isaaclab-issue-triage` | `.agent/skills/isaaclab-issue-triage/SKILL.md` | Fetch issue, validate bug template fields, route |
| `isaaclab-bug-reproduce` | `.agent/skills/isaaclab-bug-reproduce/SKILL.md` | Checkout commit, run repro steps, evaluate result |
| `isaaclab-bug-fix` | `.agent/skills/isaaclab-bug-fix/SKILL.md` | Branch, fix, test, changelog, pre-commit, PR |

### Required context files

The skills reference these repo files directly (no duplication):
- `.github/ISSUE_TEMPLATE/bug.md` — expected bug report fields
- `.github/PULL_REQUEST_TEMPLATE.md` — PR format
- `docs/source/refs/contributing.rst` — coding style and contribution process
- `AGENTS.md` (this file) — changelog, commit, naming, and testing rules

### Running for a single issue

When asked to triage a specific issue number, execute the full chain:

1. **Read** `.agent/skills/isaaclab-issue-triage/SKILL.md` — fetch and validate the issue
2. If the issue is valid for reproduction, **read** `.agent/skills/isaaclab-bug-reproduce/SKILL.md` — attempt reproduction
3. If the bug still reproduces on latest, **read** `.agent/skills/isaaclab-bug-fix/SKILL.md` — implement and submit the fix

Stop at any step that reaches a terminal state (commented, closed, or needs user input).

### Running for all open bugs

```bash
claude "Run the issue triage workflow for all open bug issues"
```

This processes each open bug issue sequentially through the full chain.

## Sandbox & Networking

- Network access (e.g., `git push`) is blocked by the sandbox. Use `dangerouslyDisableSandbox: true` so the user gets an approval prompt — don't ask them to run it manually.

## GitHub Actions and CI/CD

- IMPORTANT: Pin actions by SHA hash. Use `action@<sha>  # vX.Y.Z` format for supply-chain security. Check existing workflows in `.github/workflows/` for the allowlisted hashes. New actions or versions require repo admin approval to be added to the allowlist.

## Testing Guidelines

- **Always verify regression tests fail without the fix.** When writing a regression test for a bug fix, temporarily revert the fix and run the test to confirm it fails. Then reapply the fix and verify the test passes. This ensures the test actually covers the bug.

### Debugging Warp kernels

**Do not add `wp.printf` to kernels in production code.** Debug prints in Warp kernels affect performance and can produce noisy test output. Use them only in standalone reproduction scripts during development, and always remove them before committing.

To debug Warp kernel behavior:

1. **Write a standalone reproduction script** and run it directly with `./isaaclab.sh -p -c "..."` or `./isaaclab.sh -p script.py`. This keeps stdout visible and avoids the test framework entirely.
2. **Use high-precision format strings** for floating-point debugging (e.g., `wp.printf("val=%.15e\n", x)`) — the default `%f` format hides values smaller than ~1e-6 that can still affect control flow.
3. **Remove all `wp.printf` calls before committing.**
