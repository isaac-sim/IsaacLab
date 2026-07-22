<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# PR 6474 Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve Mike Michelis's five verified review findings while making runtime and play `num_frames` count measured calls independently of warm-up.

**Architecture:** Keep runtime warm-up as a distinct phase and keep play as one continuous rollout so reward and episode statistics span warm-up. Split play wall-time samples before runtime assembly, validate resolved training workloads before environment creation where possible, let early stopping omit unavailable timing, and make recorder wrapper installation transactional.

**Tech Stack:** Python 3.12, pytest, argparse, Isaac Lab benchmark schema/builders, uv, pre-commit.

## Global Constraints

- Use `/tmp/isaaclab-pr-6474-review/.venv`, created from the PR's locked uv project.
- Run Python and pytest through `uv run --no-sync ./isaaclab.sh -p`.
- `num_frames` is the exact measured-call count; total runtime/play calls equal warm-up plus measured calls.
- Play warm-up contributes to reward, episode-length, success, and resource statistics, but not timing or throughput.
- Training remains iteration-controlled; warm-up does not extend learning.
- Remove the obsolete play parser rejection test and replace it with positive semantic coverage.
- Verify every regression test fails on the pre-fix implementation before applying its fix.
- Run `uv run --no-sync ./isaaclab.sh -f` before every commit.
- Do not add dependencies or edit `CHANGELOG.rst` or `config/extension.toml`.

---

## File map

- `source/isaaclab/isaaclab/test/benchmark/stepping.py`: runtime/play stepping and transactional recorder wrappers.
- `source/isaaclab/isaaclab/test/benchmark/builders.py`: runtime assembly, empty early-stop timing, and effective means.
- `source/isaaclab/isaaclab/test/benchmark/schema.py`: `MeanStd` statistical contract.
- `source/isaaclab/isaaclab/test/benchmark/_cli.py`: resolved-workload warm-up validation.
- `scripts/benchmarks/runtime.py`: runtime startup-sample selection and CLI wording.
- `scripts/benchmarks/{rsl_rl,rl_games,sb3,skrl}/benchmark_*_play.py`: exact measured play workloads.
- `scripts/benchmarks/{rsl_rl,rl_games,sb3,skrl}/benchmark_*_train.py`: resolved training validation, early-stop behavior, and unconditional cleanup.
- `source/isaaclab/test/benchmark/test_stepping.py`: exact warm-up and recorder rollback regressions.
- `source/isaaclab/test/benchmark/test_builders.py`: empty timing and conventional standard-deviation regressions.
- `source/isaaclab/test/benchmark/test_cli.py`: resolved-workload validator tests.
- `scripts/benchmarks/test/test_benchmark_smoke.py`: parser and end-to-end adapter expectations.
- `scripts/benchmarks/test/test_runtime_smoke.py`: zero/default runtime workload checks.
- `docs/source/testing/benchmarks.rst`: user-facing measured/warm-up contract.
- `source/isaaclab/changelog.d/antoiner-runtime-benchmark-warmup.minor.rst`: user-visible changes and fixes.

---

### Task 1: Make runtime and play frame counts independent of warm-up

**Files:**
- Modify: `source/isaaclab/test/benchmark/test_stepping.py`
- Modify: `scripts/benchmarks/test/test_benchmark_smoke.py`
- Modify: `source/isaaclab/isaaclab/test/benchmark/stepping.py`
- Modify: `scripts/benchmarks/runtime.py`
- Modify: `scripts/benchmarks/rsl_rl/benchmark_rsl_rl_play.py`
- Modify: `scripts/benchmarks/rl_games/benchmark_rl_games_play.py`
- Modify: `scripts/benchmarks/sb3/benchmark_sb3_play.py`
- Modify: `scripts/benchmarks/skrl/benchmark_skrl_play.py`
- Modify: `docs/source/testing/benchmarks.rst`
- Modify: `source/isaaclab/changelog.d/antoiner-runtime-benchmark-warmup.minor.rst`

**Interfaces:**
- Consumes: `run_runtime_loop(env, num_frames, *, reset=True)` and `run_play_loop(env, policy, num_frames)`.
- Produces: exact warm-up behavior from `run_runtime_warmup(env, num_frames)`; play adapters with `len(measured_step_times) == num_frames`.

- [ ] **Step 1: Replace the obsolete runtime warm-up test with exact-count coverage**

```python
@pytest.mark.parametrize("num_frames", [0, 1, 50])
def test_run_runtime_warmup_runs_exact_requested_steps(num_frames: int):
    env = _Env()

    times = run_runtime_warmup(env, num_frames=num_frames)

    assert env.reset_called
    assert env.steps == num_frames
    assert len(times) == num_frames
```

- [ ] **Step 2: Remove the obsolete play rejection test and add acceptance coverage**

Delete `test_play_adapters_reject_warmup_that_exhausts_workload`. Add:

```python
@pytest.mark.parametrize("library", ["rsl_rl", "rl_games", "skrl", "sb3"])
def test_play_adapters_accept_warmup_larger_than_measured_workload(library: str, monkeypatch):
    module = _load_adapter(library, "play")
    argv = ["--task", _TASK, "--num_frames", "2", "--warmup_steps", "3", "--headless"]
    monkeypatch.setattr(sys, "argv", ["benchmark", *argv])

    args = module._parse_args(argv)[0]

    assert args.num_frames == 2
    assert args.warmup_steps == 3
```

Keep the existing end-to-end assertions of 250 host-return samples and 10 synchronized samples; they are the regression expectations for the new contract.

- [ ] **Step 3: Run the new tests against the pre-fix code**

Run:

```bash
uv run --no-sync ./isaaclab.sh -p -m pytest \
  source/isaaclab/test/benchmark/test_stepping.py::test_run_runtime_warmup_runs_exact_requested_steps \
  scripts/benchmarks/test/test_benchmark_smoke.py::test_play_adapters_accept_warmup_larger_than_measured_workload -q
```

Expected: FAIL because zero runtime warm-up executes one step and all four play parsers reject warm-up greater than `num_frames`.

- [ ] **Step 4: Implement exact runtime warm-up and startup selection**

Change `run_runtime_warmup` to:

```python
def run_runtime_warmup(env, num_frames: int) -> list[float]:
    """Run exactly ``num_frames`` excluded warm-up environment steps.

    Args:
        env: A Gym-compatible environment.
        num_frames: Number of warm-up environment steps.

    Returns:
        Per-step wall times [s] for the excluded steps.
    """
    return run_runtime_loop(env, num_frames)
```

In `runtime.py`, keep the measured loop at `args.num_frames` and select:

```python
first_step_s = warmup_step_times_s[0] if warmup_step_times_s else step_times_s[0]
```

Update `--warmup_frames` help to state that it is the exact number of excluded calls and that zero includes the first call in measurement.

- [ ] **Step 5: Implement measured play slicing in all four adapters**

Remove each parser's `warmup_steps >= num_frames` error. In each adapter use its local argument namespace (`args` for RSL-RL, `args_cli` for the other three):

```python
total_frames = args.warmup_steps + args.num_frames
with environment_step_timer, BenchmarkMonitor(benchmark, interval=1.0):
    all_step_times, reward, ep_length, success_rate = stepping.run_play_loop(env, policy, total_frames)

first_step_s = all_step_times[0]
step_times = all_step_times[args.warmup_steps :]
```

Use `first_step_s` for `StartupTime.first_step`. Use only `step_times` for FPS and `builders.build_runtime`. Retain reward, episode length, success, and resource values from the complete rollout. In RL-Games, SB3, and skrl, use the displayed block with every `args` reference replaced by that file’s `args_cli` namespace; the total-frame expression, slice boundary, and measured outputs remain identical.

Update all four `--num_frames` help strings to say "Number of measured environment steps" and all four `--warmup_steps` strings to say "Number of preceding environment steps excluded from timing and throughput."

- [ ] **Step 6: Update documentation and changelog wording**

Document these formulas in `benchmarks.rst`:

```text
runtime total calls = warmup_frames + num_frames
play total calls = warmup_steps + num_frames
```

State that play warm-up remains part of reward/episode/resource statistics. Update the existing changelog entry to say `num_frames` counts measured calls and total calls therefore increase by the requested warm-up count; existing invocations need no argument migration.

- [ ] **Step 7: Run focused tests**

Run:

```bash
uv run --no-sync ./isaaclab.sh -p -m pytest \
  source/isaaclab/test/benchmark/test_stepping.py \
  scripts/benchmarks/test/test_runtime_smoke.py \
  scripts/benchmarks/test/test_benchmark_smoke.py::test_play_adapters_accept_warmup_larger_than_measured_workload \
  scripts/benchmarks/test/test_benchmark_smoke.py::test_adapters_default_to_one_warmup_step -q
```

Expected: PASS.

- [ ] **Step 8: Run pre-commit and commit**

```bash
uv run --no-sync ./isaaclab.sh -f
git add source/isaaclab/isaaclab/test/benchmark/stepping.py \
  source/isaaclab/test/benchmark/test_stepping.py \
  scripts/benchmarks/runtime.py scripts/benchmarks/*/benchmark_*_play.py \
  scripts/benchmarks/test/test_benchmark_smoke.py docs/source/testing/benchmarks.rst \
  source/isaaclab/changelog.d/antoiner-runtime-benchmark-warmup.minor.rst
git commit -m "Fix benchmark measured frame counts"
```

---

### Task 2: Reject impossible training warm-up and close environments unconditionally

**Files:**
- Create: `source/isaaclab/test/benchmark/test_cli.py`
- Modify: `source/isaaclab/isaaclab/test/benchmark/_cli.py`
- Modify: `source/isaaclab/test/benchmark/test_builders.py`
- Modify: `source/isaaclab/isaaclab/test/benchmark/builders.py`
- Modify: `scripts/benchmarks/rsl_rl/benchmark_rsl_rl_train.py`
- Modify: `scripts/benchmarks/rl_games/benchmark_rl_games_train.py`
- Modify: `scripts/benchmarks/sb3/benchmark_sb3_train.py`
- Modify: `scripts/benchmarks/skrl/benchmark_skrl_train.py`
- Modify: `source/isaaclab/changelog.d/antoiner-runtime-benchmark-warmup.minor.rst`

**Interfaces:**
- Produces: `validate_warmup_steps(warmup_steps: int, available_steps: int) -> None`.
- Extends: `build_runtime(..., allow_empty_environment_step_timing: bool = False) -> Runtime`.

- [ ] **Step 1: Add failing validator tests**

Create `test_cli.py` with the 2026 SPDX header and:

```python
"""Tests for benchmark CLI value validation."""

import pytest

from isaaclab.test.benchmark._cli import validate_warmup_steps


@pytest.mark.parametrize(("warmup_steps", "available_steps"), [(0, 1), (15, 16)])
def test_validate_warmup_steps_accepts_a_remaining_sample(warmup_steps: int, available_steps: int):
    validate_warmup_steps(warmup_steps, available_steps)


@pytest.mark.parametrize(("warmup_steps", "available_steps"), [(1, 1), (17, 16)])
def test_validate_warmup_steps_rejects_exhausted_workload(warmup_steps: int, available_steps: int):
    with pytest.raises(ValueError, match="must be less than resolved training environment steps"):
        validate_warmup_steps(warmup_steps, available_steps)
```

- [ ] **Step 2: Add a failing early-stop empty-timing test**

Add to `test_builders.py`:

```python
def test_build_runtime_can_omit_timing_when_early_stop_exhausts_samples():
    with pytest.warns(RuntimeWarning, match="environment-step timing omitted"):
        runtime = builders.build_runtime(
            startup_time_s=StartupTime(0.1, 0.2, 0.3),
            iteration_times_s=[1.0],
            collection_fps=[8.0],
            total_fps=[8.0],
            steps_per_iteration=8,
            frames_per_environment_step=8,
            environment_step_times_s=[],
            simulation_step_times_s=[],
            simulation_step_calls=0,
            allow_empty_environment_step_timing=True,
        )

    assert runtime.environment_step_timing is None
```

- [ ] **Step 3: Run both regressions against pre-fix code**

```bash
uv run --no-sync ./isaaclab.sh -p -m pytest \
  source/isaaclab/test/benchmark/test_cli.py \
  source/isaaclab/test/benchmark/test_builders.py::test_build_runtime_can_omit_timing_when_early_stop_exhausts_samples -q
```

Expected: collection/import failure for the missing validator and `TypeError` for the missing builder keyword.

- [ ] **Step 4: Implement resolved-workload validation**

Add to `_cli.py`:

```python
def validate_warmup_steps(warmup_steps: int, available_steps: int) -> None:
    """Validate that training warm-up leaves at least one measured environment step."""
    if warmup_steps >= available_steps:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be less than resolved training environment steps "
            f"({available_steps})"
        )
```

After each adapter resolves its agent configuration and before environment creation, call it with:

```python
# RSL-RL
validate_warmup_steps(args_cli.warmup_steps, agent_cfg.max_iterations * agent_cfg.num_steps_per_env)

# RL-Games
training_steps = int(agent_cfg["params"]["config"]["max_epochs"]) * int(
    agent_cfg["params"]["config"].get("horizon_length", 16)
)
validate_warmup_steps(args_cli.warmup_steps, training_steps)

# SB3
validate_warmup_steps(args_cli.warmup_steps, resolved_max_iterations * n_steps_cfg)

# skrl
validate_warmup_steps(args_cli.warmup_steps, resolved_max_iterations * rollouts)
```

Import `validate_warmup_steps` beside `parse_non_negative_int` and `parse_positive_int` in each adapter’s lazy `_cli` import, and import it inside `run` before the resolved configuration check.

- [ ] **Step 5: Implement explicit early-stop empty timing behavior**

Add `allow_empty_environment_step_timing: bool = False` to `build_runtime`. When `environment_step_times_s` is an empty sequence and the flag is true:

```python
if not environment_samples:
    if allow_empty_environment_step_timing:
        warnings.warn(
            "environment-step timing omitted because no samples remained after warm-up",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        raise ValueError(
            "environment_step_times_s must contain only positive samples; no samples remained after warm-up, "
            "so reduce warmup_steps or increase the workload"
        )
```

Only construct `EnvironmentStepTiming` in the nonempty branch. Import `warnings`. Document that the option is for workloads that may stop before the nominal resolved length. Pass `allow_empty_environment_step_timing=True` from all four training adapters only; runtime and play retain strict validation.

- [ ] **Step 6: Make environment cleanup unconditional in all training adapters**

RSL-RL already imports `contextlib` inside `run`; add that lazy import to RL-Games, SB3, and skrl. Immediately after the final environment wrapper is installed, open `with contextlib.closing(env):` and indent the contiguous block through `benchmark._finalize_impl()`. The first indented statement is `runner_types` in RSL-RL, `runner = Runner(observer)` in RL-Games, `policy_arch = agent_cfg.pop("policy")` in SB3, and `from skrl.utils.runner.torch import Runner` in skrl. Remove the trailing `env.close()` from all four files. This ensures close runs for runner, training, parsing, builder, and formatter exceptions.

- [ ] **Step 7: Run focused tests and the original exhaustion reproduction**

```bash
uv run --no-sync ./isaaclab.sh -p -m pytest \
  source/isaaclab/test/benchmark/test_cli.py \
  source/isaaclab/test/benchmark/test_builders.py -q

uv run --no-sync ./isaaclab.sh -p scripts/benchmarks/training.py \
  --rl_library rsl_rl --task Isaac-Cartpole-Direct --num_envs 16 \
  --max_iterations 1 --warmup_steps 16 presets=newton_mjwarp --headless \
  --output_path /tmp/isaaclab-pr-6474-warmup-validation
```

Expected: unit tests PASS. The command fails before training with the resolved-workload validation message and does not create a checkpoint or benchmark bundle.

- [ ] **Step 8: Update the changelog, run pre-commit, and commit**

Add a `Fixed` entry stating that impossible training warm-up is rejected before the workload, early stopping can omit unavailable timing, and environments close on failure.

```bash
uv run --no-sync ./isaaclab.sh -f
git add source/isaaclab/isaaclab/test/benchmark/_cli.py \
  source/isaaclab/isaaclab/test/benchmark/builders.py \
  source/isaaclab/test/benchmark/test_cli.py \
  source/isaaclab/test/benchmark/test_builders.py \
  scripts/benchmarks/*/benchmark_*_train.py \
  source/isaaclab/changelog.d/antoiner-runtime-benchmark-warmup.minor.rst
git commit -m "Validate benchmark training warm-up"
```

---

### Task 3: Make recorder wrapper installation transactional

**Files:**
- Modify: `source/isaaclab/test/benchmark/test_stepping.py`
- Modify: `source/isaaclab/isaaclab/test/benchmark/stepping.py`
- Modify: `source/isaaclab/changelog.d/antoiner-runtime-benchmark-warmup.minor.rst`

**Interfaces:**
- Preserves: `EnvironmentStepTimingRecorder.__enter__()` and `__exit__()` signatures.
- Adds only private installation-state flags and a private restoration method.

- [ ] **Step 1: Add a failing installation rollback test**

Add a test environment whose class method can be read but whose instance assignment is rejected:

```python
class _ReadOnlyStepEnv(_Env):
    def __setattr__(self, name, value):
        if name == "step" and getattr(self, "_lock_step", False):
            raise AttributeError("step is read-only")
        super().__setattr__(name, value)

    def __init__(self):
        super().__init__()
        self._lock_step = True


def test_environment_step_timer_rolls_back_partial_wrapper_installation():
    env = _ReadOnlyStepEnv()
    recorder = EnvironmentStepTimingRecorder(env, measure_synchronized_step_breakdown=True)

    with pytest.raises(AttributeError, match="step is read-only"):
        recorder.__enter__()

    assert "step" not in vars(env.unwrapped.sim)
    assert recorder._original_env_step is None
    assert recorder._original_sim_step is None
```

- [ ] **Step 2: Verify the regression fails**

```bash
uv run --no-sync ./isaaclab.sh -p -m pytest \
  source/isaaclab/test/benchmark/test_stepping.py::test_environment_step_timer_rolls_back_partial_wrapper_installation -q
```

Expected: FAIL because `sim.step` remains in `vars(env.unwrapped.sim)` and recorder originals remain set.

- [ ] **Step 3: Implement transactional install and best-effort restoration**

Initialize `_env_wrapper_installed` and `_sim_wrapper_installed` to `False`. Move both assignments to the end of `__enter__` after wrapper functions are defined:

```python
try:
    if self._measure_synchronized_step_breakdown:
        assert self._simulation_context is not None
        self._simulation_context.step = timed_simulation_step
        self._sim_wrapper_installed = True
    self._env.step = timed_environment_step
    self._env_wrapper_installed = True
except BaseException:
    try:
        self._restore_wrappers()
    except BaseException:
        pass
    raise
```

Implement `_restore_wrappers()` so it restores only successfully installed wrappers, always clears both flags and both `_original_*` sentinels, and restores the simulation wrapper in a `finally` even if environment restoration raises. Replace `__exit__` body with `self._restore_wrappers()`.

- [ ] **Step 4: Run recorder tests**

```bash
uv run --no-sync ./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark/test_stepping.py -q
```

Expected: PASS.

- [ ] **Step 5: Update changelog, run pre-commit, and commit**

Add a `Fixed` entry for wrapper rollback after partial installation failure.

```bash
uv run --no-sync ./isaaclab.sh -f
git add source/isaaclab/isaaclab/test/benchmark/stepping.py \
  source/isaaclab/test/benchmark/test_stepping.py \
  source/isaaclab/changelog.d/antoiner-runtime-benchmark-warmup.minor.rst
git commit -m "Make benchmark timing wrappers exception-safe"
```

---

### Task 4: Report conventional standard deviation for effective throughput

**Files:**
- Modify: `source/isaaclab/test/benchmark/test_builders.py`
- Modify: `source/isaaclab/isaaclab/test/benchmark/builders.py`
- Modify: `source/isaaclab/isaaclab/test/benchmark/schema.py`
- Modify: `source/isaaclab/changelog.d/antoiner-runtime-benchmark-warmup.minor.rst`

**Interfaces:**
- Preserves: `MeanStd` fields and serialized schema.
- Changes: effective-throughput `std` becomes the ordinary sample standard deviation of per-sample rates.

- [ ] **Step 1: Change tests to require conventional sample standard deviation**

Import `statistics` in `test_builders.py`. Replace effective-throughput expectations with:

```python
assert rt.total_fps.std == pytest.approx(statistics.stdev([8.0, 8.0 / 3.0]))
assert rt.collection_fps.std == pytest.approx(statistics.stdev([8.0, 8.0 / 3.0]))
```

For environment-step FPS samples `[8.0, 4.0]`, use:

```python
assert rt.environment_step_timing.environment_step_fps.std == pytest.approx(statistics.stdev([8.0, 4.0]))
```

- [ ] **Step 2: Verify the changed expectations fail**

```bash
uv run --no-sync ./isaaclab.sh -p -m pytest \
  source/isaaclab/test/benchmark/test_builders.py::test_build_runtime_uses_effective_aggregate_throughput_when_requested \
  source/isaaclab/test/benchmark/test_builders.py::test_build_runtime_adds_environment_step_timing -q
```

Expected: FAIL with reported values `4.216370...` and `2.981423...` instead of conventional sample deviations.

- [ ] **Step 3: Preserve the existing sample deviation when replacing the mean**

Change the helper and its calls to:

```python
def _with_effective_mean(stats: MeanStd, effective_mean: float) -> MeanStd:
    """Return statistics with an effective aggregate mean."""
    return MeanStd(mean=effective_mean, std=stats.std, peak=stats.peak)
```

Update all calls to remove the sample-sequence argument. Update `build_runtime` and `MeanStd` docstrings to say `std` is always the ordinary sample standard deviation of the per-sample values, while `mean` may be an effective aggregate rate.

- [ ] **Step 4: Run builder, schema, and formatter tests**

```bash
uv run --no-sync ./isaaclab.sh -p -m pytest \
  source/isaaclab/test/benchmark/test_builders.py \
  source/isaaclab/test/benchmark/test_schema.py \
  source/isaaclab/test/benchmark/test_benchmark_core.py -q
```

Expected: PASS, and flat measurements remain correctly labeled `Std`.

- [ ] **Step 5: Update changelog, run pre-commit, and commit**

Add a `Fixed` entry stating effective throughput now reports conventional per-sample standard deviation.

```bash
uv run --no-sync ./isaaclab.sh -f
git add source/isaaclab/isaaclab/test/benchmark/builders.py \
  source/isaaclab/isaaclab/test/benchmark/schema.py \
  source/isaaclab/test/benchmark/test_builders.py \
  source/isaaclab/changelog.d/antoiner-runtime-benchmark-warmup.minor.rst
git commit -m "Use conventional benchmark rate deviation"
```

---

### Task 5: Full verification and review handoff

**Files:**
- Verify only; do not modify source unless a verification failure identifies a defect in Tasks 1-4.

**Interfaces:**
- Consumes all outputs from Tasks 1-4.
- Produces a clean branch with reproducible verification evidence.

- [ ] **Step 1: Run the benchmark unit suite**

```bash
uv run --no-sync ./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark -q
```

Expected: all tests PASS.

- [ ] **Step 2: Run lightweight adapter parser coverage**

```bash
uv run --no-sync ./isaaclab.sh -p -m pytest scripts/benchmarks/test/test_benchmark_smoke.py \
  -k "adapters_reject_non_positive_workloads or adapters_reject_negative_warmup_steps or adapters_default_to_one_warmup_step or adapters_accept_short_synchronized_step_flag or play_adapters_accept_warmup_larger_than_measured_workload" -q
```

Expected: all selected tests PASS.

- [ ] **Step 3: Run the exact RSL-RL end-to-end smoke regression**

```bash
uv run --no-sync ./isaaclab.sh -p -m pytest scripts/benchmarks/test/test_benchmark_smoke.py \
  -k "test_training_and_play_write_bundles and rsl_rl" -vv -x
```

Expected: PASS with 250 host-return samples and 10 synchronized samples.

- [ ] **Step 4: Run documentation and changelog checks**

```bash
uv run --no-sync ./isaaclab.sh -d
uv run --no-sync ./isaaclab.sh -p tools/changelog/cli.py check develop
```

Expected: both commands PASS.

- [ ] **Step 5: Run final pre-commit and clean-state verification**

```bash
uv run --no-sync ./isaaclab.sh -f
git diff --check
git status --short --branch
```

Expected: every hook PASS, `git diff --check` prints nothing, and the branch has no uncommitted source changes.

- [ ] **Step 6: Prepare the review response summary**

Report each Mike thread as fixed with its commit and verification evidence. Do not post GitHub replies or push until the user explicitly requests those external actions.
