# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Control-period recording of joint targets, as consumed by a target-forwarding deployment."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("physicsnemo_robotics")

import torch
from isaaclab_physicsnemo import JOINT_TARGET_KEY, NewtonBinding, TrajectoryDatasetRecorder
from physicsnemo_robotics.data import NewtonTrajectoryZarrReader
from physicsnemo_robotics.newton import field_to_torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim import SimulationContext

# The ANYmal-C scenes are those of the example scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "anymal_c"))
from anymal_c_env_cfg import AnymalCExplorationEnvCfg, AnymalCTargetControlEnvCfg  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _env(cfg):
    cfg.scene.num_envs = 2
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset(seed=0)
    return env


def test_recorded_targets_match_the_deployed_control(tmp_path):
    """The recorded control of a 20 ms period is exactly what the learned scene hands the model for that action."""
    actions = torch.linspace(-1.0, 1.0, 12).repeat(2, 1)
    teacher = _env(AnymalCExplorationEnvCfg())
    teacher_identity = NewtonBinding(teacher.sim.physics_manager).spec.model.fingerprint
    with TrajectoryDatasetRecorder(
        teacher, tmp_path, scenario_id="test", control_targets=teacher.scene["robot"], replace=True
    ) as recorder:
        frame_dt = recorder.schema.simulation.time.frame_dt
        for _ in range(2):
            teacher.step(actions.to(teacher.device))
            recorder.record(torch.zeros(2, dtype=torch.bool, device=teacher.device))
    teacher.close()
    SimulationContext.clear_instance()

    learned = _env(AnymalCTargetControlEnvCfg())
    binding = NewtonBinding(learned.sim.physics_manager)
    learned.step(actions.to(learned.device))
    deployed = field_to_torch(binding.control.joint_target_q).reshape(2, -1).cpu()
    learned.close()
    SimulationContext.clear_instance()
    with NewtonTrajectoryZarrReader(tmp_path) as reader:
        recorded = reader.read_slice(0, 0, 1, (JOINT_TARGET_KEY,))[JOINT_TARGET_KEY][0].cpu()

    # One frame is one control period of four 5 ms physics steps, in both scenes.
    assert frame_dt == pytest.approx(0.02) and learned.step_dt == pytest.approx(0.02)
    # Zero-gain target forwarding leaves the Newton model, and so the artifact identity, unchanged.
    assert binding.spec.model.fingerprint == teacher_identity
    torch.testing.assert_close(recorded, deployed[0], atol=1e-6, rtol=0.0)
