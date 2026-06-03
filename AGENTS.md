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

**CRITICAL: Always run pre-commit hooks BEFORE committing and BEFORE pushing.**

Proper workflow:
1. Make your code changes
2. Run `./isaaclab.sh -f` to check ALL files
3. If pre-commit modifies any files (e.g., formatting), review the changes
4. Stage the modified files with `git add`
5. Run `./isaaclab.sh -f` again to ensure all checks pass
6. Only then create your commit with `git commit`
7. Verify pre-commit still passes before pushing — never push commits that haven't been checked

```bash
# Run pre-commit checks on all files
./isaaclab.sh -f
```

**Common mistakes to avoid:**
- Don't commit first and then run pre-commit (requires amending commits)
- Don't push before running pre-commit (pushes broken code to the remote)
- Do run pre-commit before committing and before pushing (clean workflow)

**When reviewing code** (e.g. via a code-reviewer agent), always run `./isaaclab.sh -f` as part of the review to catch formatting or lint issues early.

## Changelog

- **Do not edit `CHANGELOG.rst` or `config/extension.toml` directly.** Each PR adds a fragment file under `source/<package>/changelog.d/`; the changelog and version are compiled by the nightly CI workflow.
- **Add one fragment per touched package.** Pick any short, unique slug for the filename — your branch name (with `/` replaced by `-`) is a good default. The filename suffix declares the bump tier; within a batch the highest tier wins for the package.

  | Filename | Effect |
  |---|---|
  | `source/<pkg>/changelog.d/<slug>.rst` | patch bump |
  | `source/<pkg>/changelog.d/<slug>.minor.rst` | minor bump |
  | `source/<pkg>/changelog.d/<slug>.major.rst` | major bump |
  | `source/<pkg>/changelog.d/<slug>.skip` | no entry, no bump (CI / docs / test-only) |

- Use **past tense** matching the section header: "Added X", "Fixed Y", "Changed Z".
- Place entries under the correct category: `Added`, `Changed`, `Deprecated`, `Removed`, or `Fixed`.
- Avoid internal implementation details users wouldn't understand.
- **For `Deprecated`, `Changed`, and `Removed` entries, include migration guidance.**
  - Example: "Deprecated `Articulation.A` in favor of `Articulation.B`."
- **Breaking changes** belong in `Changed`, prefixed with `**Breaking:**`.
- Use Sphinx cross-reference roles for class/method/module names.

### RST formatting reference

```
Added
^^^^^

* Added :class:`~package.ClassName` to support feature X.

Fixed
^^^^^

* Fixed edge case in :meth:`~package.ClassName.method` where input was
  not validated, causing ``AttributeError`` at runtime.
```

Key formatting rules:
- Category heading: underline with `^` (carets), at least as long as the heading text.
- Entries: `* ` prefix, continuation lines indented by 2 spaces.

See `tools/changelog/test/integration/` for worked examples that double as integration-test fixtures.

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

## Sandbox & Networking

- Network access (e.g., `git push`) is blocked by the sandbox. Use `dangerouslyDisableSandbox: true` so the user gets an approval prompt — don't ask them to run it manually.
- **Never push to `origin` (`isaac-sim/IsaacLab`).** The `origin` remote is the public upstream repository. Push to your own fork remote (e.g., `antoine`, `alex`) or to the remote of the PR you are working on. If the correct remote is unclear, ask the user before pushing.

## GitHub Actions and CI/CD

- Pin actions by major version tag (e.g. `actions/checkout@v6`). Use the same major version that other workflows in `.github/workflows/` already use — don't introduce a new major version without checking how it's used elsewhere.

## Testing Guidelines

- **Always verify regression tests fail without the fix.** When writing a regression test for a bug fix, temporarily revert the fix and run the test to confirm it fails. Then reapply the fix and verify the test passes. This ensures the test actually covers the bug.

### Install CI tests (`source/isaaclab/test/install_ci/`)

These tests exist to validate documented installation paths end-to-end. Follow the rules below exactly. If a rule conflicts with what you want to do, push back to the maintainer rather than deviating.

#### Scope: installation integration only

- **Belongs here:** tests that create a fresh env (uv/conda/system Python), run an install command (`./isaaclab.sh -i <args>`, `uv pip install <wheel>[extras]`), and verify imports / smoke training in the resulting env.
- **Does not belong here:** unit tests on `pyproject.toml` / `python_packages.toml` contents, unit tests on the install CLI's argument parsing, anything that does not actually install. Those go to `source/isaaclab/test/cli/`.
- **Self-containment:** files under `install_ci/` MUST NOT reference paths outside the directory. `(cd source/isaaclab/test/install_ci && pytest)` must work after copying the directory elsewhere.
- **Wheel comes from the runner, not the test.** Tests under `uv_pip/` MUST NOT call `tools/wheel_builder/build.sh` themselves; instead they consume the `wheel` session fixture (errors out if no wheel was provided). The runner builds it on demand via `tools/run_install_ci.py --build-wheel` or accepts a pre-built one via `--wheel <path>`, exposed to pytest through `ISAACLAB_WHEEL`.

#### Directory layout

- `cli/` — exercises `./isaaclab.sh -i <args>`.
- `uv_pip/` — exercises `uv pip install <wheel>[extras]` against a built wheel.
- `misc/` — wheel-builder smoke, env-marker unit tests, anything else install-adjacent.

#### Filename convention (strict)

- `cli/test_cli_install_<install_args>_in_<envname>_<modifier>.py`
- `uv_pip/test_uv_pip_install_<details>.py`
- `misc/test_<descriptive>_smoke.py`

Allowed `<modifier>` values:
- `_smoke` — quick sanity checks (env creation, Python version, single import).
- `_correctness` — verifies the install scope is exactly right (nothing extra, nothing missing).
- `_training` — runs the full cartpole training end-to-end.
- `_runs_training` — installs an extra (RL framework, visualizer) and runs training to verify the extra.

Notes:
- If the install scope is empty (`./isaaclab.sh -i` with no args), drop it: `test_cli_install_in_<env>_<modifier>.py`.
- The env suffix MUST sort before the modifier: `..._in_uvenv_smoke.py`, never `..._smoke_in_uvenv.py`.

#### Class naming (strict)

Snake-style `Test_<Filename_Stem_TitleCase>`, mirroring the filename. Each underscore-separated segment is TitleCase. Common acronyms stay uppercase when separate segments. **Do not** use CamelCase here; snake-style is the deliberate convention for these long compound names.

- ✅ `Test_Cli_Install_Core_In_Uvenv_Correctness`
- ✅ `Test_Uv_Pip_Install_Isaaclab_All_Isaacsim_Trains_Cartpole`
- ❌ `TestCliInstallCoreInUvenvCorrectness`

#### Method naming (strict)

`test_<install_cmd>_<outcome>` — each method reads as "test that running X produces Y".

- ✅ `test_install_core_makes_assets_importable`
- ✅ `test_install_all_trains_cartpole`
- ✅ `test_install_rl_pulls_all_frameworks`
- ❌ `test_core_install_assets` (no outcome verb)
- ❌ `test_install_global` (no outcome)

For single-test files, keep the method name long and descriptive. Do not shorten because the class already encodes the scope; more tests may be added.

#### Top docstring (strict)

Every test file's module docstring uses this exact `Setup:` / `Tests:` format:

```
"""
Setup:
    - <full shell command run before tests>
Tests:
    - <full command> -> verify <outcome>
    - <full command> -> verify <outcome>
"""
```

Rules:
- Spell every command in full. **Never** abbreviate with `...`.
- Wrap long commands with continuation indent (no `\` line continuations).
- `Setup:` lists ONLY commands that actually run before tests (env creation, wheel build, install). Do not pad with documentation snippets the test does not execute.
- `Tests:` lists one entry per test method, in declaration order.

#### Setup steps must match the docs

The `Setup:` block must mirror the install commands a user would run from the official installation docs at `docs/source/setup/installation/` (e.g. `binaries_installation.rst`, `source_installation.rst`, `pip_installation.rst`, `uv_run.rst`).

- If the docs and a test diverge, fix one of them and call the change out in the PR description.
- Do not invent install commands the docs do not endorse. `install_ci/` validates the documented install paths, not a parallel one.
- When adding a new install path that the docs do not yet cover, update the docs in the same PR.

#### Markers (strict)

- `@pytest.mark.docker` xor `@pytest.mark.native` — never both. Omit when the test is environment-agnostic.
- `@pytest.mark.uv` / `@pytest.mark.conda` — gate on the package manager when required.
- `@pytest.mark.gpu` — requires a GPU.
- `@pytest.mark.slow` — long-running (use whenever an install runs).
- `@pytest.mark.install_path_cli` — test exercises the `./isaaclab.sh -i` install path. Apply to every test under `cli/`.
- `@pytest.mark.install_path_uv_pip` — test exercises the `uv pip install <wheel>` install path. Apply to every test under `uv_pip/` (class-level decorator is cleanest).
- `@pytest.mark.timeout(<seconds>)` — set explicitly per test, sized to the slowest valid run.

#### isaacsim availability gating

Tests that import isaacsim or run training must gate with the per-test helper, not class-level `setup_class`:

```python
def _skip_if_isaacsim_unavailable() -> None:
    """Skip the current test when isaacsim is neither importable nor symlinked at ``_isaac_sim``."""
    try:
        import isaacsim  # noqa: F401
    except ImportError:
        if not (find_isaaclab_root() / "_isaac_sim").exists():
            pytest.skip("isaacsim is not importable and _isaac_sim link not found, skipping")
```

Per-test gating keeps lightweight smoke tests (Python version, env creation) from being skipped when they do not need isaacsim.

#### Training assertion helper (mandatory)

Training tests reuse a shared `_assert_training_passed(result)` that asserts:

1. `result.returncode == 0`
2. `Traceback (most recent call last):` does NOT appear in combined stdout/stderr.
3. `Training time:` DOES appear in combined stdout/stderr.

Do not re-invent these assertions per file.

#### Standard cartpole training command

```python
_TRAIN_CMD = [
    "train",
    "--rl_library", "rsl_rl",
    "--task", "Isaac-Cartpole-Direct",
    "--num_envs", "16",
    "presets=newton_mjwarp",
    "--max_iterations", "5",
    "--headless",
]
```

Invoke via `self.run_in_uv_env([str(self.cli_script)] + _TRAIN_CMD, ...)` (or the conda equivalent). This is the canonical "did training run" probe; do not redefine the args list per file.

### Debugging Warp kernels

**Do not add `wp.printf` to kernels in production code.** Debug prints in Warp kernels affect performance and can produce noisy test output. Use them only in standalone reproduction scripts during development, and always remove them before committing.

To debug Warp kernel behavior:

1. **Write a standalone reproduction script** and run it directly with `./isaaclab.sh -p -c "..."` or `./isaaclab.sh -p script.py`. This keeps stdout visible and avoids the test framework entirely.
2. **Use high-precision format strings** for floating-point debugging (e.g., `wp.printf("val=%.15e\n", x)`) — the default `%f` format hides values smaller than ~1e-6 that can still affect control flow.
3. **Remove all `wp.printf` calls before committing.**
