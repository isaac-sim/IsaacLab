# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for ``MJWarpNeRDSolver`` on the Allegro cube reorientation task."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("newton_neural", reason="needs the newton-neural extra")

import newton
import torch
import warp as wp
from newton_neural.models import Precision
from newton_neural.solvers import SolverNeRDMuJoCo

from isaaclab.test.utils import DeviceScope, test_devices

from nerd_test_utils import (  # isort: skip
    ALLEGRO_REORIENT,
    WORLD_COUNT,
    environment,
    nerd_task_cfg,
    notify_change,
    record_and_train,
    refused,
    save_point_model,
    task_cfg,
)

pytestmark = [pytest.mark.integration, pytest.mark.kitless]

TASK = ALLEGRO_REORIENT


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The model of the hand's free motion, recorded and trained by the scripts once for all tests of this module."""
    return record_and_train(TASK, tmp_path_factory.mktemp("allegro"), num_envs=128, num_frames=128, max_steps=50)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_free_motion_task(checkpoint: Path, device: str, caplog: pytest.LogCaptureFixture):
    """Run the cube task on MuJoCo Warp with the model for the hand, and expect the task's settings and resets."""
    reference = task_cfg(TASK, WORLD_COUNT, device)
    mujoco = reference.sim.physics.solver_cfg
    env_cfg = nerd_task_cfg(TASK, checkpoint, WORLD_COUNT, device)
    with environment(TASK, env_cfg, info=True) as env:
        backend = env.sim.backend_manager.backend
        solver = backend.solver
        assert isinstance(solver, SolverNeRDMuJoCo)
        # The task keeps its timing, and a solver step of MuJoCo Warp is the frame of the model.
        timing = (env.physics_dt, env.cfg.decimation, backend.num_substeps)
        assert timing == (reference.sim.dt, reference.decimation, reference.sim.physics.num_substeps)
        assert backend.solver_dt == pytest.approx(solver.nerd.frame_dt, rel=1e-6)
        # The solver class logs what it built, which ``--info`` shows. The step of the network is compiled.
        built = [entry.getMessage() for entry in caplog.records if entry.name.startswith("isaaclab_neural")]
        assert built == [f"Built {solver}"]
        assert solver.stepper.compiled
        # The settings of the task's MuJoCo Warp reach the solver.
        assert solver.mj_model.opt.impratio == mujoco.impratio

        env.reset()
        for _ in range(100):
            action = 2.0 * torch.rand(WORLD_COUNT, env.action_manager.total_action_dim, device=device) - 1.0
            observations, reward, _, _, _ = env.step(action)
            assert observations["policy"].isfinite().all()
            assert reward.isfinite().all()
        assert wp.to_torch(backend.state_0.joint_q).isfinite().all()
        assert wp.to_torch(backend.state_0.joint_qd).isfinite().all()
        assert list(backend.graphs) == [("simulate", reference.decimation)]

        # A reset of some environments clears the history of the model in exactly those worlds.
        env.sim.forward()  # applies the resets that the last step left pending
        before = solver.stepper.export_history()
        restarted = torch.arange(WORLD_COUNT, device=device) % 4 == 0
        env.reset(torch.nonzero(restarted).flatten())
        for name, value in solver.stepper.export_history().items():
            assert not value[restarted].any(), name
            assert torch.equal(value[~restarted], before[name][~restarted]), name
            assert before[name][~restarted].any(), name

        # A change of the masses reaches MuJoCo Warp and not the network, for which the solver warns once.
        masses = newton.ModelFlags.BODY_INERTIAL_PROPERTIES
        assert dict(backend.solver_ops.ignored_model_changes) == {}
        assert notify_change(env, masses, caplog) == [SolverNeRDMuJoCo.UNSEEN_MODEL_CHANGES[masses]]
        assert notify_change(env, masses, caplog) == []


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_solver_settings(checkpoint: Path, device: str):
    """Expect the precision and the compilation that the configuration asks for in the step of the network."""
    env_cfg = nerd_task_cfg(TASK, checkpoint, 4, device, precision="fp32", compile=False)
    with environment(TASK, env_cfg) as env:
        stepper = env.sim.backend_manager.backend.solver.stepper
        assert (stepper.precision, stepper.compiled) == (Precision.FP32, False)
        env.reset()
        for _ in range(8):
            action = torch.zeros(4, env.action_manager.total_action_dim, device=device)
            observations, *_ = env.step(action)
        assert observations["policy"].isfinite().all()


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_free_motion_task_without_triton(checkpoint: Path, device: str, monkeypatch: pytest.MonkeyPatch):
    """Without Triton, which PyTorch installs on Linux only, the solver steps the model without ``torch.compile``."""
    monkeypatch.setitem(sys.modules, "triton", None)  # Importing Triton fails, as where it is not installed
    with environment(TASK, nerd_task_cfg(TASK, checkpoint, 4, device)) as env:
        env.reset()
        action = torch.zeros(4, env.action_manager.total_action_dim, device=device)
        observations, *_ = env.step(action)
        assert observations["policy"].isfinite().all()
        assert not env.sim.backend_manager.backend.solver.stepper.compiled


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_refusals(checkpoint: Path, device: str, tmp_path: Path):
    """Expect a task that does not fit the model to be refused with the configuration to change."""
    # A physics step whose solver steps are not frames of the model.
    env_cfg = nerd_task_cfg(TASK, checkpoint, 4, device)
    env_cfg.sim.dt = 0.01
    with environment(TASK, env_cfg) as env:
        env.reset()
        with pytest.raises(ValueError, match="model was trained on frames"):
            env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=device))

    # A model of another kind of system than a robot, which is refused before a solver is built on it.
    env_cfg = nerd_task_cfg(TASK, save_point_model(tmp_path / "point.pt"), 4, device)
    refused(TASK, env_cfg, TypeError, "is a SystemPoint, and an Isaac Lab task runs on models of articulated")
