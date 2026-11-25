# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for ``NeRDSolver`` under the ordinary Newton manager on the ANYmal C flat walking task."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("newton_neural", reason="needs the newton-neural extra")

import newton
import torch
from isaaclab_neural.nerd import build_model
from isaaclab_newton.physics import NewtonManager
from newton_neural import Frames
from newton_neural.data import TrajectoryDataset
from newton_neural.models import Precision
from newton_neural.solvers import SolverNeRD

from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ImuCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.stage import use_stage
from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_tasks.contrib.velocity.config.anymal_c.flat_env_cfg import ANYMAL_C_NERD_TARGET_REACH
from isaaclab_tasks.utils import load_cfg_from_registry

from nerd_test_utils import (  # isort: skip
    ANYMAL_C_FLAT,
    WORLD_COUNT,
    environment,
    nerd_task_cfg,
    notify_change,
    record_and_train,
    refused,
    recorded_warnings,
    save_point_model,
    simulation,
    task_cfg,
    unseen_changes,
)

pytestmark = [pytest.mark.integration, pytest.mark.kitless]

TASK = ANYMAL_C_FLAT


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The model of the robot, recorded and trained by the scripts once for all tests of this module."""
    return record_and_train(TASK, tmp_path_factory.mktemp("anymal_c"), num_envs=64, num_frames=128, max_steps=100)


def _random_actions(env: Any) -> torch.Tensor:
    """Actions of the walking task drawn from a standard normal distribution."""
    return torch.randn(env.num_envs, env.action_manager.total_action_dim, device=env.device)


def _state(solver: SolverNeRD, state: newton.State) -> Frames:
    """A copy of the state signals of the robots in ``state``."""
    return {name: value.clone() for name, value in solver.binding.read_state(state).items()}


@pytest.mark.parametrize("device", test_devices())
def test_nerd_task(checkpoint: Path, device: str, caplog: pytest.LogCaptureFixture):
    """Run the walking task on the model, captured on CUDA, and expect the timing of the task and the model's steps."""
    captured = torch.device(device).type == "cuda"
    mujoco = task_cfg(TASK, WORLD_COUNT, device)
    env_cfg = nerd_task_cfg(TASK, checkpoint, WORLD_COUNT, device)
    env_cfg.sim.use_newton_actuators = captured
    solver_cfg, legs = env_cfg.sim.physics.solver_cfg, env_cfg.scene.robot.actuators["legs"]
    with recorded_warnings() as caught, environment(TASK, env_cfg, info=True) as env:
        manager = env.sim.backend_manager
        backend = manager.backend
        solver = backend.solver
        assert type(manager) is NewtonManager
        assert isinstance(solver, SolverNeRD)
        nerd, binding, stepper = solver.nerd, solver.binding, solver.stepper

        # Physics timing is explicit; the task control period stays equal to the numerical reference.
        assert env.cfg.decimation == 1
        assert env.physics_dt == pytest.approx(mujoco.sim.dt * mujoco.decimation)
        assert env.step_dt == pytest.approx(nerd.frame_dt, rel=1e-6)
        assert backend.dt == backend.solver_dt == nerd.frame_dt
        # The solver takes the precision of the preset and compiles the step on CUDA. It logs what it built,
        # which ``--info`` shows.
        assert (stepper.precision, stepper.compiled) == (Precision(solver_cfg.precision), captured)
        built = [entry.getMessage() for entry in caplog.records if entry.name.startswith("isaaclab_neural")]
        assert built == [f"Built {solver}"]
        # The contact sensor of the scene reads detected contacts, and the model stands for the actuators.
        assert backend.collision_pipeline is not None
        assert not backend.model.actuators
        # The preset randomizes nothing at startup that the model does not see.
        assert unseen_changes(caught) == []

        # Random actions through the step of the task. Each one is one physics step and one control period.
        env.reset()
        actions = env.action_manager.get_term("joint_pos")
        step_count = 100 if captured else 20
        first_step, start = env.sim.get_physics_step_count(), manager.get_simulation_time()
        for _ in range(step_count):
            observations, _, terminated, truncated, _ = env.step(_random_actions(env))
            kept = ~(terminated | truncated)
            # The solver reads the joint targets of the actions.
            targets = binding.read_control(backend.control)["joint_target_q"]
            assert torch.equal(targets[kept], actions.processed_actions[kept])
            assert observations["policy"].isfinite().all()
        assert env.sim.get_physics_step_count() - first_step == step_count
        assert manager.get_simulation_time() - start == pytest.approx(step_count * env.step_dt)
        assert list(backend.graphs) == ([("simulate", 1)] if captured else [])

        # A reset of some environments clears the history of the model in exactly those worlds.
        env.sim.forward()  # applies the resets that the last step left pending
        before = stepper.export_history()
        restarted = torch.arange(WORLD_COUNT, device=device) % 4 == 0
        env.reset(torch.nonzero(restarted).flatten())
        for name, value in stepper.export_history().items():
            assert not value[restarted].any(), name
            assert torch.equal(value[~restarted], before[name][~restarted]), name
            assert before[name][~restarted].any(), name

        # Each step of the task is one step of the eager model from the same frame, to the rounding of the
        # precision. The two next states are compared as the labels of the network they came from. The joint
        # efforts that Isaac Lab reports are those of a PD controller with the gains of the preset.
        env.reset()
        sensor = nerd.contact_input.sensor(backend.model, binding)
        robot = env.scene["robot"]
        compared = torch.ones(WORLD_COUNT, dtype=torch.bool, device=device)
        for _ in range(6):
            env.sim.forward()
            frame = _state(solver, backend.state_0)
            frame["contact_probes"] = sensor.update(backend.state_0, None)["contact_probes"].clone()
            history = stepper.export_history()
            _, _, terminated, truncated, _ = env.step(_random_actions(env))
            compared &= ~(terminated | truncated)
            frame["joint_target_q"] = actions.processed_actions.clone()
            with torch.no_grad():
                expected = nerd.labels(frame, nerd.step(frame, history))[compared]
            stepped = nerd.labels(frame, _state(solver, backend.state_0))[compared]
            assert (stepped - expected).abs().max().item() < 2e-2 * max(1.0, expected.abs().max().item())
            effort = legs.stiffness * (frame["joint_target_q"] - frame["joint_q"][:, 7:])
            effort -= legs.damping * frame["joint_qd"][:, 6:]
            limit = legs.actuator_effort_limit
            reported = robot.actuators.applied_effort.torch
            assert torch.allclose(reported[compared], effort.clamp(-limit, limit)[compared], atol=1e-3)
        assert compared.sum() > WORLD_COUNT // 2

        # Actions far outside the range that the model was recorded on leave the target of every joint within it,
        # which is the default pose plus and minus the task's shared target reach.
        env.reset()
        reach = ANYMAL_C_NERD_TARGET_REACH
        far = torch.full_like(actions.raw_actions, 100.0)
        far[:, ::2] = -100.0
        _, _, terminated, truncated, _ = env.step(far)
        kept = ~(terminated | truncated)
        assert kept.any()
        bound = robot.data.default_joint_pos.torch + reach * far.sign()
        targets = binding.read_control(backend.control)["joint_target_q"]
        assert torch.allclose(targets[kept], bound[kept], atol=1e-6)

        # With the feet below the ground, the contact sensor reports the force estimates of the solver for them.
        env.reset()
        sunk = robot.data.root_link_pose_w.torch.clone()
        sunk[:, 2] -= 0.25
        robot.write_root_link_pose_to_sim_mask(root_pose=sunk)
        robot.write_joint_position_to_sim_mask(position=robot.data.default_joint_pos.torch.clone())
        env.sim.forward()
        assert (sensor.update(backend.state_0, None)["contact_probes"][..., 0] < 0).all()
        _, _, terminated, truncated, _ = env.step(torch.zeros_like(actions.raw_actions))
        kept = ~(terminated | truncated)
        assert kept.any()
        contact_forces = env.scene["contact_forces"]
        feet, _ = contact_forces.find_sensors(".*FOOT")
        forces = contact_forces.data.net_forces_w.torch[kept][:, feet]
        assert (forces.norm(dim=-1) > 0).all()
        assert forces.isfinite().all()

        # A change of the Newton model does not reach the network. The solver warns once for each kind of
        # change. The solver class declares no ignored changes, for which Isaac Lab would log a warning itself.
        flags, unseen = newton.ModelFlags, SolverNeRD.UNSEEN_MODEL_CHANGES
        assert dict(backend.solver_ops.ignored_model_changes) == {}
        for change in (flags.BODY_INERTIAL_PROPERTIES, flags.MODEL_PROPERTIES):
            assert notify_change(env, change, caplog) == [unseen[change]]
            assert notify_change(env, change, caplog) == []

        # A hard reset of the simulation replaces the Newton model and the solver. The manager keeps the decimation
        # that the environment declared, and the task runs on.
        env.sim.reset()
        backend = env.sim.backend_manager.backend
        assert isinstance(backend.solver, SolverNeRD)
        assert backend.solver is not solver
        env.reset()
        for _ in range(4):
            observations, *_ = env.step(_random_actions(env))
        assert observations["policy"].isfinite().all()
        assert list(backend.graphs) == ([("simulate", 1)] if captured else [])


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_nerd_task_without_contact_sensors(checkpoint: Path, device: str):
    """Run without contact sensors using the standard Newton contact lifecycle and explicit solver settings."""
    env_cfg = nerd_task_cfg(TASK, checkpoint, 4, device, precision="fp32", compile=False)
    env_cfg.scene.contact_forces = None
    env_cfg.rewards.feet_air_time = env_cfg.rewards.undesired_contacts = env_cfg.terminations.base_contact = None
    # Without explicit pipeline settings, Newton constructs its default collision pipeline.
    env_cfg.sim.physics.collision_cfg = None
    with environment(TASK, env_cfg) as env:
        backend = env.sim.backend_manager.backend
        env.reset()
        for _ in range(8):
            observations, *_ = env.step(_random_actions(env))
        assert observations["policy"].isfinite().all()
        assert backend.collision_pipeline is not None
        assert backend.contacts is not None
        # The solver runs the step in the precision and without the compilation that the configuration asks for.
        stepper = backend.solver.stepper
        assert (stepper.precision, stepper.compiled) == (Precision.FP32, False)
        assert list(backend.graphs) == [("simulate", 1)]


# Windows CI runs this where Triton is really missing, with the model that the scripts record and train there.
@pytest.mark.windows_ci
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_nerd_task_without_triton(checkpoint: Path, device: str, monkeypatch: pytest.MonkeyPatch):
    """Without Triton, which PyTorch installs on Linux only, the solver steps the model without ``torch.compile``."""
    monkeypatch.setitem(sys.modules, "triton", None)  # Importing Triton fails, as where it is not installed
    with environment(TASK, nerd_task_cfg(TASK, checkpoint, 4, device)) as env:
        env.reset()
        observations, *_ = env.step(_random_actions(env))
        assert observations["policy"].isfinite().all()
        assert not env.sim.backend_manager.backend.solver.stepper.compiled


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_step_without_an_environment(checkpoint: Path, device: str):
    """A standalone simulation uses its declared timestep without an environment-specific timing handshake."""
    env_cfg = nerd_task_cfg(TASK, checkpoint, 4, device, compile=False)
    with simulation(env_cfg):
        sim = SimulationContext(env_cfg.sim)
        with use_stage(sim.stage):
            sim.register_interactive_scene(InteractiveScene(env_cfg.scene))
            sim.reset()
        manager = sim.backend_manager
        # A step of a paused simulation advances nothing, so it is no mistake.
        sim.pause()
        sim.step()
        sim.play()
        first_step = sim.get_physics_step_count()
        sim.step()
        assert sim.get_physics_step_count() == first_step + 1
        assert manager.get_simulation_time() == pytest.approx(env_cfg.sim.dt)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_refusals(checkpoint: Path, device: str, tmp_path: Path):
    """Expect a task that does not fit the model to be refused with the configuration to change."""
    # Native stepping rejects a timestep that differs from the checkpoint's model frame.
    env_cfg = nerd_task_cfg(TASK, checkpoint, 4, device)
    env_cfg.sim.dt /= 2
    with environment(TASK, env_cfg) as env:
        env.reset()
        with pytest.raises(ValueError, match="model was trained on frames"):
            env.step(_random_actions(env))

    # The physics of the model, with the actuator network of the task still running.
    env_cfg = nerd_task_cfg(TASK, checkpoint, 4, device)
    env_cfg.scene.robot.actuators = task_cfg(TASK, 4, device).scene.robot.actuators
    refused(TASK, env_cfg, ValueError, "ImplicitActuatorCfg")

    # A robot that is not where the configuration says.
    env_cfg = nerd_task_cfg(TASK, checkpoint, 4, device, articulation="{ENV_REGEX_NS}/Arm")
    refused(TASK, env_cfg, ValueError, r"env_cfg\.scene\.robot\.prim_path")

    # A sensor that reads accelerations and a robot that reads joint reaction forces. A model that predicts the
    # next state computes neither.
    env_cfg = nerd_task_cfg(TASK, checkpoint, 4, device)
    env_cfg.scene.imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base")
    env_cfg.scene.robot.enable_joint_wrench = True
    refused(TASK, env_cfg, ValueError, r"body_qdd \(.*\) and body_parent_f \(")

    # A model that was built on the dataset and never fitted, which has no frame to step the task by.
    dataset = TrajectoryDataset(checkpoint.parent / "dataset")
    build_model(dataset, load_cfg_from_registry(TASK, "nerd_cfg_entry_point")).save(tmp_path / "unfitted.pt")
    env_cfg = nerd_task_cfg(TASK, tmp_path / "unfitted.pt", 4, device)
    refused(TASK, env_cfg, ValueError, "no frame duration.*Fit its normalization")

    # A task drives the joints and the actuators of a robot, so a model of another kind of system is refused, and
    # that before a solver is built on it, which the system of this model does not allow.
    env_cfg = nerd_task_cfg(TASK, save_point_model(tmp_path / "point.pt"), 4, device)
    refused(TASK, env_cfg, TypeError, "is a SystemPoint, and an Isaac Lab task runs on models of articulated")


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_checkpoint_contract_mismatches(checkpoint, device, tmp_path):
    """Reject a checkpoint whose recorded robot properties differ from the built Newton model."""
    from newton_neural.models import NeRDBase

    nerd = NeRDBase.load(checkpoint, device="cpu")
    nerd.metadata["isaaclab"]["physical_model"]["body_mass"][0] += 1.0
    wrong_mass = tmp_path / "wrong_mass.pt"
    nerd.save(wrong_mass)
    refused(TASK, nerd_task_cfg(TASK, wrong_mass, 4, device), ValueError, "physical model differs")
