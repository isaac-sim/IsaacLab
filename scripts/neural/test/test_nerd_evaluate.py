# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ``scripts/neural/evaluate.py``, which scores policies in the physics that a model learned from."""

from __future__ import annotations

import math
import re
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("newton_neural", reason="needs the newton-neural extra")
pytest.importorskip("rsl_rl", reason="needs the rsl-rl extra")

import torch
from isaaclab_newton.physics import NewtonCfg
from rsl_rl.runners import OnPolicyRunner

from isaaclab.managers import CommandTerm, CommandTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.sim import SimulationContext
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import to_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.utils import load_cfg_from_registry, resolve_task_config

from scripts.neural import evaluate

# The helpers of the ``isaaclab_neural`` tests resolve the tasks and launch the simulation.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "source" / "isaaclab_neural" / "test"))

from nerd_test_utils import (  # noqa: E402
    ANYMAL_C_FLAT,
    FRANKA_REACH,
    TASKS,
    cuda_device,
    environment,
    nerd_preset,
    task_cfg,
)

pytestmark = [pytest.mark.integration, pytest.mark.kitless]

AGENT = "rsl_rl_cfg_entry_point"
RECIPE = "nerd_recipe_cfg_entry_point"
UNSEEN_EVENTS = {ANYMAL_C_FLAT: ("add_base_mass", "base_com")}
"""Events that the recipe of a task names as unseen by its model. The other tasks name none."""
WORLD_COUNT = 8
EPISODE_STEPS = 8
"""Environment steps that a policy is scored for."""
EARLY_STEPS = 3
"""Environment steps after which the episodes of the first half of the environments end."""
LATE_STEPS = 6
"""Environment steps after which the episodes of the third quarter of the environments end."""


def _one_per_second(env: Any) -> torch.Tensor:
    """A reward of one, which the reward manager weighs by the duration of a step."""
    return torch.ones(env.num_envs, device=env.device)


def _early(env: Any) -> torch.Tensor:
    """End the episodes of the first half of the environments after :data:`EARLY_STEPS` steps."""
    index = torch.arange(env.num_envs, device=env.device)
    return (index < env.num_envs // 2) & (env.episode_length_buf >= EARLY_STEPS)


def _late(env: Any) -> torch.Tensor:
    """End the episodes of the third quarter of the environments after :data:`LATE_STEPS` steps."""
    index = torch.arange(env.num_envs, device=env.device)
    third_quarter = (index >= env.num_envs // 2) & (index < 3 * env.num_envs // 4)
    return third_quarter & (env.episode_length_buf >= LATE_STEPS)


class _IndexCommand(CommandTerm):
    """A command that asks for nothing, whose one metric is the index of the environment."""

    def __init__(self, cfg: CommandTermCfg, env: Any) -> None:
        super().__init__(cfg, env)
        self.metrics["index"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 1, device=self.device)

    def _update_metrics(self) -> None:
        self.metrics["index"][:] = torch.arange(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: Sequence[int] | slice) -> None:
        pass

    def _update_command(self) -> None:
        pass


class _RestingPolicy:
    """Zero actions, with the reset by which a recurrent policy forgets the episodes that ended."""

    def __init__(self, actions: torch.Tensor) -> None:
        self.actions = actions
        self.forgotten = 0
        """Episodes that the policy was told to forget."""

    def __call__(self, observations: Any) -> torch.Tensor:
        return self.actions

    def reset(self, dones: torch.Tensor) -> None:
        self.forgotten += int(dones.sum())


class _Launched(Exception):
    """Raised in place of the launch of a simulation."""


def _save_run(run: Path, env_cfg: Any, policy: Path | None = None, iteration: int = 0) -> Path:
    """Lay out the folder ``run`` as a training run saves it, with the settings of its task and a policy file.

    Without ``policy`` the file is empty, which is enough for a run that is only looked up.
    """
    dump_yaml(str(run / "params" / "env.yaml"), env_cfg)
    file = run / f"model_{iteration}.pt"
    if policy is None:
        file.touch()
    else:
        shutil.copy(policy, file)
    return file


def _launched_cfg(task: str, folder: Path, monkeypatch: pytest.MonkeyPatch, *overrides: str) -> Any:
    """Return the configuration that ``evaluate.py`` launches the simulation of ``task`` with, without launching it.

    The working directory becomes ``folder``, which receives a run of the task on MuJoCo Warp for the script to find.
    """
    monkeypatch.chdir(folder)
    env_cfg, agent_cfg = resolve_task_config(task, AGENT, overrides=("physics=newton_mjwarp",))
    _save_run(folder / "logs" / "rsl_rl" / agent_cfg.experiment_name / "2026-01-01_08-00-00", env_cfg)
    launched = []

    def launch(cfg: Any, args: Any) -> None:
        launched.append(cfg)
        raise _Launched

    monkeypatch.setattr(evaluate, "launch_simulation", launch)
    with pytest.raises(_Launched):
        evaluate.main(["--task", task, "--num_envs", "4", *overrides, "--trained_on", "newton_mjwarp"])
    (env_cfg,) = launched
    return env_cfg


@pytest.fixture(scope="module")
def saved_policy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An untrained policy of the reach task, in the file that a training run saves."""
    device = cuda_device()
    env_cfg, agent_cfg = resolve_task_config(FRANKA_REACH, AGENT, overrides=("physics=newton_mjwarp",))
    env_cfg.scene.num_envs, env_cfg.sim.device, env_cfg.seed = 2, device, 0
    file = tmp_path_factory.mktemp("policy") / "model_0.pt"
    with environment(FRANKA_REACH, env_cfg) as env:
        OnPolicyRunner(RslRlVecEnvWrapper(env), to_dict(agent_cfg), log_dir=None, device=device).save(str(file))
    return file


def test_trained_run(tmp_path: Path):
    """Among runs that share a folder, expect the latest one that saved a policy on the wanted physics."""
    presets = (nerd_preset(FRANKA_REACH), "newton_mjwarp", "newton_steel")
    env_cfgs = {name: resolve_task_config(FRANKA_REACH, None, overrides=(f"physics={name}",))[0] for name in presets}
    physics = {name: evaluate.physics_classes(to_dict(env_cfg.sim)["physics"]) for name, env_cfg in env_cfgs.items()}
    model, mujoco, steel = presets
    # Runs are named by the time they started, and each saves the settings of its task as the train script does.
    runs = {
        "2026-01-01_08-00-00": model,
        "2026-01-02_08-00-00": mujoco,
        "2026-01-03_08-00-00": model,
        "2026-01-04_08-00-00": mujoco,
    }
    for run, preset in runs.items():
        _save_run(tmp_path / run, env_cfgs[preset])
    # A run that ended before it saved a policy is passed over, and so is one that saved no settings.
    dump_yaml(str(tmp_path / "2026-01-05_08-00-00" / "params" / "env.yaml"), env_cfgs[model])
    (tmp_path / "2026-01-06_08-00-00").mkdir()
    (tmp_path / "2026-01-06_08-00-00" / "model_0.pt").touch()
    # So is a run whose settings name no physics, as a task without a physics configuration saves them.
    env_cfgs[steel].sim.physics = None
    _save_run(tmp_path / "2026-01-07_08-00-00", env_cfgs[steel])

    assert evaluate.trained_run(tmp_path, physics[model]) == tmp_path / "2026-01-03_08-00-00"
    assert evaluate.trained_run(tmp_path, physics[mujoco]) == tmp_path / "2026-01-04_08-00-00"
    assert evaluate.trained_run(tmp_path, physics[steel]) is None


def test_main(saved_policy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """Evaluate the reach task from a folder of runs, and expect the score of the policy of each chosen run."""
    device = cuda_device()
    monkeypatch.chdir(tmp_path)
    presets = ("newton_mjwarp", "newton_steel")
    cfgs = {name: resolve_task_config(FRANKA_REACH, AGENT, overrides=(f"physics={name}",)) for name in presets}
    (mujoco, agent_cfg), (steel, _) = cfgs.values()
    folder = tmp_path / "logs" / "rsl_rl" / agent_cfg.experiment_name
    # Two runs that started in the same second share the folder. The name of the later one starts with the name
    # of the run on MuJoCo Warp, and it trained longer on other physics. Both saved the same policy.
    trained = _save_run(folder / "2026-01-01_08-00-00", mujoco, saved_policy)
    trained_on_steel = _save_run(folder / "2026-01-01_08-00-00_steel", steel, saved_policy, iteration=5)

    argv = ["--task", FRANKA_REACH, "--num_envs", "2", "--device", device, "env.episode_length_s=0.1"]
    try:
        evaluate.main([*argv, "--trained_on", nerd_preset(FRANKA_REACH), *presets])
    finally:
        SimulationContext.clear_instance()

    printed = capsys.readouterr().out
    # Each policy comes from exactly the run on the wanted physics, and no run trained on the model.
    assert f"Policy trained with physics=newton_mjwarp, {trained}" in printed
    assert f"Policy trained with physics=newton_steel, {trained_on_steel}" in printed
    assert f"No policy trained with physics={nerd_preset(FRANKA_REACH)} yet" in printed
    assert "on average over the first episode of 2 environments" in printed
    # Every policy starts from the same seed, so the same policy scores the same at any place in the list.
    scores = re.findall(r"^  On physics=newton_mjwarp an episode collects a reward of .*$", printed, flags=re.MULTILINE)
    metrics = re.findall(r"^  \S+ is \S+ on average over the episodes\.$", printed, flags=re.MULTILINE)
    assert len(scores) == 2 and scores[0] == scores[1]
    assert len(metrics) >= 2 and metrics[: len(metrics) // 2] == metrics[len(metrics) // 2 :]


@pytest.mark.parametrize("task", TASKS)
def test_main_runs_the_robot_of_the_model(task: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Expect the simulation to be launched on the play settings of the task, without the events its model does not see.

    Nothing else differs. What the recipe changes for the recording stays as the task has it, such as the table of
    the arm and the constraint rows of the hand.
    """
    env_cfg = _launched_cfg(task, tmp_path, monkeypatch)

    play, _ = resolve_task_config(task, AGENT, play_mode=True, overrides=("physics=newton_mjwarp",))
    play.scene.num_envs, play.seed = 4, 0
    # The play settings keep the events that the model does not see, and the evaluation turns them off.
    for name in UNSEEN_EVENTS.get(task, ()):
        assert getattr(play.events, name) is not None, name
        setattr(play.events, name, None)
    assert to_dict(env_cfg) == to_dict(play)


def test_main_runs_on_the_named_physics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Expect the policies to run on the physics that the command line names, in either form, Newton or not."""
    physx = type(resolve_task_config(FRANKA_REACH, None, overrides=("physics=physx",))[0].sim.physics)
    assert not issubclass(physx, NewtonCfg)
    for selector in ("physics", "env.sim.physics"):
        env_cfg = _launched_cfg(FRANKA_REACH, tmp_path, monkeypatch, f"{selector}=physx")
        assert type(env_cfg.sim.physics) is physx, selector


def test_request_mistakes_end_before_the_simulation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """A request that cannot be evaluated ends with its reason on the command line, before a simulation starts."""
    monkeypatch.chdir(tmp_path)
    requests = {
        # A preset that the task does not offer, which the error names.
        "newton_stel": ["--task", FRANKA_REACH, "--trained_on", "newton_stel"],
        f"no policy of {FRANKA_REACH} to evaluate": ["--task", FRANKA_REACH],
    }
    for reason, argv in requests.items():
        with pytest.raises(SystemExit):
            evaluate.main(argv)
        captured = capsys.readouterr()
        assert reason in captured.err
        assert SimulationContext.instance() is None
    # The last request names no preset, so the script looked for a policy of the model preset and for one of the
    # preset that the model learned from.
    for preset in (nerd_preset(FRANKA_REACH), "newton_mjwarp"):
        assert f"No policy trained with physics={preset} yet" in captured.out

    # An event that the recipe names as unseen by the model and the task does not have, here a misspelled one.
    recipe_cfg = load_cfg_from_registry(ANYMAL_C_FLAT, RECIPE)
    recipe_cfg.unseen_events = ("add_base_mass", "base_mass")
    monkeypatch.setattr(evaluate, "load_cfg_from_registry", lambda task, entry_point: recipe_cfg)
    with pytest.raises(SystemExit):
        evaluate.main(["--task", ANYMAL_C_FLAT])
    assert "names base_mass in unseen_events" in capsys.readouterr().err
    assert SimulationContext.instance() is None


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_score(device: str):
    """Score a policy on a task whose episodes end at known steps, and expect the first episode of each to count."""
    env_cfg = task_cfg(FRANKA_REACH, WORLD_COUNT, device)
    env_cfg.episode_length_s = (EPISODE_STEPS - 0.5) * env_cfg.sim.dt * env_cfg.decimation
    # One reward per second, so that the reward of an episode is its length.
    for term in vars(env_cfg.rewards).values():
        if isinstance(term, RewardTermCfg):
            term.weight = 0.0
    env_cfg.rewards.one_per_second = RewardTermCfg(func=_one_per_second, weight=1.0)
    # Episodes end early, late or not at all, and never by reaching the goal or at the time limit.
    env_cfg.rewards.success = env_cfg.terminations.success = env_cfg.terminations.time_out = None
    env_cfg.terminations.early = TerminationTermCfg(func=_early)
    env_cfg.terminations.late = TerminationTermCfg(func=_late)
    env_cfg.commands.index = CommandTermCfg(class_type=_IndexCommand, resampling_time_range=(1.0e6, 1.0e6))

    with environment(FRANKA_REACH, env_cfg) as env:
        assert env.max_episode_length == EPISODE_STEPS
        policy = _RestingPolicy(torch.zeros(WORLD_COUNT, env.action_manager.total_action_dim, device=device))
        episode_count, reward, seconds, endings, metrics = evaluate.score(RslRlVecEnvWrapper(env), policy)
        step_dt = env.step_dt

    # The first half of the environments ends twice within the steps, the third quarter once and the rest never.
    # An environment counts once, with its first episode.
    early, late = WORLD_COUNT // 2, WORLD_COUNT // 4
    assert episode_count == early + late
    assert endings == {"early": early / episode_count, "late": late / episode_count}
    expected = (early * EARLY_STEPS + late * LATE_STEPS) * step_dt / episode_count
    assert seconds == pytest.approx(expected, rel=1e-5)
    assert reward == pytest.approx(expected, rel=1e-5)
    # The task logs a metric as its mean over the environments that reset in a step. With the early ones it logs
    # the mean of their indices. With the late ones the early ones reset again, so it logs the mean over both, which
    # counts for the late ones alone.
    with_early = sum(range(early)) / early
    with_late = sum(range(early + late)) / (early + late)
    assert metrics["index/index"] == pytest.approx((early * with_early + late * with_late) / episode_count)
    # The reach command logs how far the hand is from its goal, which a resting arm does not reach.
    assert metrics["ee_pose/position_error"] > 0.0
    assert all(math.isfinite(value) for value in metrics.values())
    # The policy is told of every episode that ended, so that a recurrent one starts each of them without memory.
    assert policy.forgotten == 2 * early + late


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_score_without_an_ended_episode(device: str):
    """Score a policy on a task whose episodes never end, and expect no episode and no number for the averages."""
    env_cfg = task_cfg(FRANKA_REACH, 2, device)
    # Two steps, and nothing that ends an episode, not even the time limit.
    env_cfg.episode_length_s = 1.5 * env_cfg.sim.dt * env_cfg.decimation
    env_cfg.rewards.success = env_cfg.terminations.success = env_cfg.terminations.time_out = None

    with environment(FRANKA_REACH, env_cfg) as env:
        actions = torch.zeros(2, env.action_manager.total_action_dim, device=device)
        episode_count, reward, seconds, endings, metrics = evaluate.score(RslRlVecEnvWrapper(env), lambda obs: actions)

    assert episode_count == 0
    assert math.isnan(reward) and math.isnan(seconds)
    assert endings == {} and metrics == {}
