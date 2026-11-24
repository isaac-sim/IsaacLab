# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ``scripts/neural/record.py``, which records the data of the model of a task."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("newton_neural", reason="needs the newton-neural extra")

import numpy as np
import torch
import yaml
from isaaclab_newton.physics import MJWarpSolverCfg
from newton_neural import Frames, SignalRole
from newton_neural.data import TrajectoryDataset
from newton_neural.models import NeRDBase

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import instantiate
from isaaclab.utils.io import load_yaml

from scripts.neural import record

# The helpers of the ``isaaclab_neural`` tests resolve the tasks and launch the simulation.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "source" / "isaaclab_neural" / "test"))

from nerd_test_utils import (  # noqa: E402
    ALLEGRO_REORIENT,
    ANYMAL_C_FLAT,
    cuda_device,
    nerd_preset,
    record_dataset,
    simulation,
    task_cfg,
)

pytestmark = [pytest.mark.integration, pytest.mark.kitless]

WORLD_COUNT = 16
FRAME_COUNT = 32
ANYMAL_C_ROUGH = "IsaacContrib-Velocity-Rough-AnymalC"
"""A task that registers no recipe."""


@pytest.mark.parametrize(("task", "two_way"), [(ANYMAL_C_FLAT, False), (ALLEGRO_REORIENT, True)])
def test_dataset(task: str, two_way: bool, tmp_path: Path):
    """Record a task and expect the signals of its model, in frames of the step that the model predicts."""
    device = cuda_device()
    folder = tmp_path / "dataset"
    record_dataset(task, folder, WORLD_COUNT, FRAME_COUNT, device)
    dataset = TrajectoryDataset(folder)

    # The untrained model lies in the dataset under the name of the file that the preset of the model loads.
    preset = task_cfg(task, WORLD_COUNT, device, overrides=(f"physics={nerd_preset(task)}",))
    nerd = NeRDBase.load(folder / Path(preset.sim.physics.solver_cfg.checkpoint).name)
    assert not nerd.fitted
    assert nerd.prediction.integrates is not two_way
    assert dataset.signals == nerd.signals
    assert (dataset.world_count, dataset.frame_count) == (WORLD_COUNT, FRAME_COUNT)

    # A model in place of the solver predicts one control period, and a model inside MuJoCo Warp one solver step.
    reference = task_cfg(task, WORLD_COUNT, device)
    sim = reference.sim
    period = sim.dt / sim.physics.num_substeps if two_way else sim.dt * reference.decimation
    assert dataset.frame_dt == pytest.approx(period, rel=1e-6)
    labels = [name for name, spec in dataset.signals.items() if spec.role is SignalRole.LABEL]
    assert labels == (["joint_qdd_free"] if two_way else [])

    arrays = dataset.arrays(0)
    assert arrays["valid"].mean() > 0.9
    for name in dataset.signals:
        assert np.isfinite(arrays[name][arrays["valid"]]).all(), name
        assert arrays[name].any(), name

    # The settings that the data was recorded with lie next to it. Those of the recipe show the sizes that the
    # command line gave, and those of the environment the robots and the frame of the data.
    sizes = load_yaml(str(folder / "params" / "recipe.yaml"))["record"]
    assert (sizes["num_envs"], sizes["num_frames"]) == (WORLD_COUNT, FRAME_COUNT)
    with open(folder / "params" / "env.yaml") as file:
        env = yaml.unsafe_load(file)
    assert env["scene"]["num_envs"] == WORLD_COUNT
    assert env["sim"]["dt"] * env["decimation"] == pytest.approx(dataset.frame_dt, rel=1e-6)


def _state(recording: record.EnvSimulation) -> Frames:
    """A copy of the state signals of the robots of ``recording``."""
    return {name: value.clone() for name, value in recording.binding.read_state(recording.state).items()}


def test_recording_simulation():
    """Restart and step the recording environment of the walking task, and expect the written states and targets."""
    device = cuda_device()
    env_cfg, recipe_cfg, _ = record.resolve_task(ANYMAL_C_FLAT, [])
    env_cfg.scene.num_envs, env_cfg.sim.device, env_cfg.seed = WORLD_COUNT, device, 0
    instantiate(recipe_cfg).configure(env_cfg)
    cfg = record.recording_cfg(env_cfg, recipe_cfg.robot, single_solver_step=False)
    with simulation(cfg):
        env = ManagerBasedEnv(cfg)
        try:
            recording = record.EnvSimulation(env, recipe_cfg.robot)
            robot = env.scene[recipe_cfg.robot]
            assert recording.world_count == WORLD_COUNT
            assert recording.frame_dt == env.step_dt

            # A restart of some environments writes their state and leaves every other environment as it is.
            before = _state(recording)
            frames = _state(recording)
            frames["joint_q"][:, :3] += torch.rand(WORLD_COUNT, 3, device=device)
            frames["joint_q"][:, 7:] += 0.2 * torch.randn(WORLD_COUNT, 12, device=device)
            frames["joint_qd"] = torch.randn(WORLD_COUNT, 18, device=device)
            restarted = torch.arange(WORLD_COUNT, device=device) % 2 == 0
            recording.restart(frames, restarted)
            written = _state(recording)
            for name in ("joint_q", "joint_qd"):
                assert torch.equal(written[name][~restarted], before[name][~restarted]), name
                assert torch.allclose(written[name][restarted], frames[name][restarted], atol=1e-6), name
            # Isaac Lab's own view of the robot follows.
            assert torch.allclose(robot.data.joint_pos.torch[restarted], frames["joint_q"][restarted, 7:], atol=1e-6)
            assert torch.allclose(robot.data.root_pos_w.torch[restarted], frames["joint_q"][restarted, :3], atol=1e-6)

            # A step lasts one frame. Robots that are let go in the air at rest fall for that long, and the solver
            # consumes the commanded joint targets, bit for bit.
            airborne = _state(recording)
            airborne["joint_q"][:, 2] += 1.0
            airborne["joint_qd"].zero_()
            recording.restart(airborne)
            held = airborne["joint_q"][:, 7:].clone()
            recording.step({"joint_target_q": held})
            assert torch.equal(recording.binding.read_control(recording.control)["joint_target_q"], held)
            fallen = _state(recording)
            fall = torch.tensor(-9.81 * recording.frame_dt, device=device)
            assert torch.allclose(fallen["joint_qd"][:, 2], fall, rtol=0.1)

            # A step without a command keeps the last targets.
            targets = held + 0.5 * torch.randn_like(held)
            recording.step({"joint_target_q": targets})
            recording.step()
            assert torch.equal(recording.binding.read_control(recording.control)["joint_target_q"], targets)

            # A restart clears what the scene keeps of a robot, such as the memory of its actuator network. The
            # same step from the same state then repeats, whatever the robots did before.
            recording.restart(airborne)
            recording.step({"joint_target_q": held})
            repeated = _state(recording)
            for name in ("joint_q", "joint_qd"):
                assert torch.allclose(repeated[name], fallen[name], atol=1e-4), name
        finally:
            env.close()


def test_recording_keeps_startup_events_only():
    """The recording environment keeps the events that set the task up and drops those that move the robots.

    The task resolves without the events that its recipe names as unseen by the model. A task without events is
    recorded with an event configuration that holds none.
    """
    env_cfg, recipe_cfg, _ = record.resolve_task(ANYMAL_C_FLAT, [])
    # The model stands for the nominal robot, so the mass and the centre of mass of the base are not randomized.
    # The event that sets the friction stays, since the model is recorded with it.
    assert env_cfg.events.add_base_mass is None and env_cfg.events.base_com is None
    assert env_cfg.events.physics_material is not None
    modes = {name: term.mode for name, term in vars(env_cfg.events).items() if isinstance(term, EventTermCfg)}
    assert {"startup", "reset", "interval"} <= set(modes.values())
    cfg = record.recording_cfg(env_cfg, recipe_cfg.robot, single_solver_step=False)
    kept = {name for name, term in vars(cfg.events).items() if isinstance(term, EventTermCfg)}
    assert kept == {name for name, mode in modes.items() if mode == "startup"}

    env_cfg.events = None
    events = record.recording_cfg(env_cfg, recipe_cfg.robot, single_solver_step=False).events
    assert events is not None and vars(events) == {}


def test_reference_preset_selects_the_recorded_physics():
    """The recipe names the physics that the model learns from, and the command line can name another.

    The settings of the recipe, which are saved beside the data, name the physics that is recorded.
    """
    steel = type(task_cfg(ANYMAL_C_FLAT, 1, "cpu", overrides=("physics=newton_steel",)).sim.physics.solver_cfg)
    assert steel is not MJWarpSolverCfg
    solvers = {"newton_mjwarp": MJWarpSolverCfg, "newton_steel": steel}
    requests = {
        (): "newton_mjwarp",
        # A setting of the recipe on the command line changes the physics that is recorded with it.
        ("agent.reference_preset=newton_steel",): "newton_steel",
        # A physics preset on the command line, in either form, goes before the recipe.
        ("physics=newton_steel",): "newton_steel",
        ("env.sim.physics=newton_steel",): "newton_steel",
        ("agent.reference_preset=newton_steel", "physics=newton_mjwarp"): "newton_mjwarp",
    }
    for overrides, preset in requests.items():
        env_cfg, recipe_cfg, _ = record.resolve_task(ANYMAL_C_FLAT, list(overrides))
        assert recipe_cfg.reference_preset == preset, overrides
        assert type(env_cfg.sim.physics.solver_cfg) is solvers[preset], overrides


def test_request_mistakes_end_before_the_simulation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """A request that cannot be recorded ends with its reason on the command line, before a simulation starts."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "meta.json").touch()
    requests = {
        "registers no recipe for a neural robot dynamics model": ["--task", ANYMAL_C_ROUGH],
        "a model is recorded from Newton physics": ["--task", ANYMAL_C_FLAT, "physics=physx"],
        # The physics of the model itself is no source of its data.
        "a model is recorded from a solver": ["--task", ANYMAL_C_FLAT, f"physics={nerd_preset(ANYMAL_C_FLAT)}"],
        # A model inside MuJoCo Warp learns the free acceleration that MuJoCo Warp computes, from no other solver.
        "learns it from MuJoCo Warp": ["--task", ALLEGRO_REORIENT, "physics=newton_steel"],
        # An event that the recipe names as unseen by the model and the task does not have.
        "names no_such_event in unseen_events": ["--task", ANYMAL_C_FLAT, "agent.unseen_events=['no_such_event']"],
        # Names without quotes reach the recipe as one text.
        "is a list of event names": ["--task", ANYMAL_C_FLAT, "agent.unseen_events=[add_base_mass]"],
        f"{tmp_path} holds a dataset": ["--task", ANYMAL_C_FLAT, "--dataset", str(tmp_path)],
    }
    for reason, argv in requests.items():
        with pytest.raises(SystemExit):
            record.main(argv)
        assert reason in capsys.readouterr().err
        assert SimulationContext.instance() is None
    assert [path.name for path in tmp_path.iterdir()] == ["meta.json"]
