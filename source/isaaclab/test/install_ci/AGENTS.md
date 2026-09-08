# Installation CI tests

## Scope

- These tests validate documented installation paths end to end.
- Tests must create a fresh environment, run an installation command, and verify imports or smoke training.
- Put unit tests for package metadata or install CLI argument parsing in `source/isaaclab/test/cli/`, not here.
- Keep this directory self-contained: copying it elsewhere and running `pytest` from it must work.
- Tests under `uv_pip/` must use the `wheel` session fixture. They must not build a wheel themselves.

## Layout and naming

- Put `./isaaclab.sh -i` tests in `cli/`, wheel-install tests in `uv_pip/`, and other installation-adjacent checks in `misc/`.
- Name CLI tests `test_cli_install_<install_args>_in_<envname>_<modifier>.py`; omit `<install_args>` when empty.
- Name wheel-install tests `test_uv_pip_install_<details>.py` and other tests `test_<description>_smoke.py`.
- Use only `smoke`, `correctness`, `training`, or `runs_training` as CLI test modifiers.
- Name test classes `Test_<Filename_Stem_TitleCase>` and test methods `test_<install_command>_<outcome>`.
- Keep common acronyms uppercase in class names and do not use CamelCase class names.

## Test contents

- Every test file has a module docstring with `Setup:` and `Tests:` sections.
- `Setup:` lists only commands that actually run before tests. `Tests:` has one full command per test method in declaration order.
- List every setup and test command in full; do not abbreviate commands or use shell line continuations.
- Keep setup commands aligned with the installation documentation in `docs/source/setup/installation/`.
- Update the documentation in the same change when adding an installation path.
- Use exactly one of `@pytest.mark.docker` or `@pytest.mark.native` when either applies.
- Use `@pytest.mark.uv` or `@pytest.mark.conda` when a package manager is required, `@pytest.mark.gpu` when a GPU is required, and `@pytest.mark.slow` whenever an installation runs.
- Apply `@pytest.mark.install_path_cli` to CLI tests and `@pytest.mark.install_path_uv_pip` to wheel-install tests.
- Set an explicit `@pytest.mark.timeout(<seconds>)` for every test.

## Isaac Sim and training

- Tests that import Isaac Sim or run training must skip per test when Isaac Sim is neither importable nor available through `_isaac_sim`.
- Do not skip lightweight tests that do not need Isaac Sim.
- Reuse the shared `_assert_training_passed(result)` helper for training assertions.
- That helper must require a zero return code, no traceback, and `Training time:` in combined output.
- Use the shared canonical Cartpole training command: RSL-RL, `Isaac-Cartpole-Direct`, 16 environments, `presets=newton_mjwarp`, and 5 iterations.
- Invoke the shared command through `self.run_in_uv_env([str(self.cli_script)] + _TRAIN_CMD, ...)` or the conda equivalent.
- Run this command headlessly without `--visualizer` or `--viz` unless a future configuration requires `--viz none`.
