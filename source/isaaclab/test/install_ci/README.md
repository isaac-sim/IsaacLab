# Install CI tests

These tests exist to validate documented installation paths end-to-end. Follow the rules below exactly. If a rule conflicts with what you want to do, push back to the maintainer rather than deviating.

## Scope: installation integration only

- **Belongs here:** tests that create a fresh env (uv/conda/system Python), run an install command (`./isaaclab.sh -i <args>`, `uv pip install <wheel>[extras]`), and verify imports / smoke training in the resulting env.
- **Does not belong here:** unit tests on `pyproject.toml` / `python_packages.toml` contents, unit tests on the install CLI's argument parsing, anything that does not actually install. Those go to `source/isaaclab/test/cli/`.
- **Self-containment:** files under `install_ci/` MUST NOT reference paths outside the directory. `(cd source/isaaclab/test/install_ci && pytest)` must work after copying the directory elsewhere.
- **Wheel comes from the runner, not the test.** Tests under `uv_pip/` MUST NOT call `tools/wheel_builder/build.sh` themselves; instead they consume the `wheel` session fixture (errors out if no wheel was provided). The runner builds it on demand via `tools/run_install_ci.py --build-wheel` or accepts a pre-built one via `--wheel <path>`, exposed to pytest through `ISAACLAB_WHEEL`.

## Directory layout

- `cli/` — exercises `./isaaclab.sh -i <args>`.
- `uv_pip/` — exercises `uv pip install <wheel>[extras]` against a built wheel.
- `misc/` — wheel-builder smoke, env-marker unit tests, anything else install-adjacent.

## Filename convention (strict)

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

## Class naming (strict)

Snake-style `Test_<Filename_Stem_TitleCase>`, mirroring the filename. Each underscore-separated segment is TitleCase. Common acronyms stay uppercase when separate segments. **Do not** use CamelCase here; snake-style is the deliberate convention for these long compound names.

- ✅ `Test_Cli_Install_Core_In_Uvenv_Correctness`
- ✅ `Test_Uv_Pip_Install_Isaaclab_All_Isaacsim_Trains_Cartpole`
- ❌ `TestCliInstallCoreInUvenvCorrectness`

## Method naming (strict)

`test_<install_cmd>_<outcome>` — each method reads as "test that running X produces Y".

- ✅ `test_install_core_makes_assets_importable`
- ✅ `test_install_all_trains_cartpole`
- ✅ `test_install_rl_pulls_all_frameworks`
- ❌ `test_core_install_assets` (no outcome verb)
- ❌ `test_install_global` (no outcome)

For single-test files, keep the method name long and descriptive. Do not shorten because the class already encodes the scope; more tests may be added.

## Top docstring (strict)

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

## Setup steps must match the docs

The `Setup:` block must mirror the install commands a user would run from the official installation docs at `docs/source/setup/installation/` (e.g. `binaries_installation.rst`, `source_installation.rst`, `pip_installation.rst`, `uv_run.rst`).

- If the docs and a test diverge, fix one of them and call the change out in the PR description.
- Do not invent install commands the docs do not endorse. `install_ci/` validates the documented install paths, not a parallel one.
- When adding a new install path that the docs do not yet cover, update the docs in the same PR.

## Markers (strict)

- `@pytest.mark.docker` xor `@pytest.mark.native` — never both. Omit when the test is environment-agnostic.
- `@pytest.mark.uv` / `@pytest.mark.conda` — gate on the package manager when required.
- `@pytest.mark.gpu` — requires a GPU.
- `@pytest.mark.slow` — long-running (use whenever an install runs).
- `@pytest.mark.install_path_cli` — test exercises the `./isaaclab.sh -i` install path. Apply to every test under `cli/`.
- `@pytest.mark.install_path_uv_pip` — test exercises the `uv pip install <wheel>` install path. Apply to every test under `uv_pip/` (class-level decorator is cleanest).
- `@pytest.mark.timeout(<seconds>)` — set explicitly per test, sized to the slowest valid run.

## isaacsim availability gating

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

## Training assertion helper (mandatory)

Training tests reuse a shared `_assert_training_passed(result)` that asserts:

1. `result.returncode == 0`
2. `Traceback (most recent call last):` does NOT appear in combined stdout/stderr.
3. `Training time:` DOES appear in combined stdout/stderr.

Do not re-invent these assertions per file.

## Standard cartpole training command

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
