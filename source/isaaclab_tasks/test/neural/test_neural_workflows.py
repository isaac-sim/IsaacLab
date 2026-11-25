# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task declarations, workflow composition and dependency ownership."""

import dataclasses
import subprocess
import sys

import pytest
from isaaclab_neural import NeRDCfg, NeRDTrainingCfg
from isaaclab_neural.record.recording_cfg import NeRDRecorderCfg

from isaaclab.utils import to_dict

from isaaclab_tasks.utils import load_cfg_from_registry, resolve_task_config
from isaaclab_tasks.utils.hydra import PresetTarget, enumerate_task_presets

TASKS = ("Isaac-Reach-Franka", "IsaacContrib-Velocity-Flat-AnymalC", "Isaac-Reorient-Cube-Allegro")
pytestmark = [pytest.mark.unit, pytest.mark.kitless]


@pytest.mark.parametrize("task", TASKS)
def test_recording_model_and_optimizer_are_independent(task):
    env, model = resolve_task_config(
        task,
        "nerd_cfg_entry_point",
        env_cfg_entry_point="nerd_recording_cfg_entry_point",
        overrides=["agent.backbone_spans=[1,1]", "env.scene.num_envs=7"],
    )
    assert env.scene.num_envs == 7
    assert tuple(model.backbone_spans) == (1, 1)
    assert isinstance(model, NeRDCfg)
    assert not {"record", "training", "train", "checkpoint", "reference_preset"} & {
        f.name for f in dataclasses.fields(model)
    }
    _, training = resolve_task_config(task, "nerd_training_cfg_entry_point", overrides=["agent.learning_rate=0.007"])
    assert isinstance(training, NeRDTrainingCfg)
    assert training.learning_rate == 0.007
    unchanged = load_cfg_from_registry(task, "nerd_cfg_entry_point")
    assert tuple(unchanged.backbone_spans) != (1, 1)
    assert env.actions.joint_targets.episode_frames[0] >= 1
    assert isinstance(env.recorders, NeRDRecorderCfg)
    # ManagerBase prepares terms only for truthy configurations, including zero-group observations.
    assert env.observations


def test_neural_solvers_use_shared_physics_declarations():
    from isaaclab_tasks.core.reach.config.franka import franka_reach_env_cfg
    from isaaclab_tasks.core.reach.reach_env_cfg import ReachPhysicsCfg
    from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_common import PhysicsCfg
    from isaaclab_tasks.core.velocity.velocity_env_cfg import RoughPhysicsCfg

    assert "newton_nerd" in RoughPhysicsCfg.__configclass_own_fields__
    assert "newton_mjwarp_nerd" in PhysicsCfg.__configclass_own_fields__
    assert "newton_nerd" in ReachPhysicsCfg.__configclass_own_fields__
    assert not hasattr(franka_reach_env_cfg, "FrankaReachPhysicsCfg")
    for task in (TASKS[0], "Isaac-Reach-Franka-OSC", "Isaac-Reach-UR10"):
        assert "newton_nerd" in enumerate_task_presets(task)[PresetTarget.PHYSICS]
    env, _ = resolve_task_config(TASKS[0], None, overrides=["physics=newton_nerd"])
    assert env.sim.physics.solver_cfg.checkpoint == "logs/neural/reach_franka/flashnerd.pt"
    assert env.sim.dt == pytest.approx(1 / 30)
    assert env.decimation == env.sim.physics.num_substeps == env.sim.render_interval == 1


@pytest.mark.parametrize(
    ("task", "selections"),
    [(TASKS[0], ("presets=newton_nerd", "presets=diffik", "presets=hold")), (TASKS[1], ("presets=newton_nerd",))],
)
def test_recording_is_taken_on_the_reference_solver(task, selections):
    """A domain or neural preset of the task changes nothing about what is recorded; a numerical solver does."""
    entry = "nerd_recording_cfg_entry_point"
    reference, _ = resolve_task_config(task, None, env_cfg_entry_point=entry, overrides=[])
    for selection in selections:
        selected, _ = resolve_task_config(task, None, env_cfg_entry_point=entry, overrides=[selection])
        assert to_dict(selected) == to_dict(reference), selection
    numerical, _ = resolve_task_config(task, None, overrides=["physics=newton_mjwarp"])
    assert reference.sim.dt == numerical.sim.dt
    assert reference.decimation == numerical.decimation
    assert to_dict(reference.scene.robot.actuators) == to_dict(numerical.scene.robot.actuators)
    steel, _ = resolve_task_config(task, None, env_cfg_entry_point=entry, overrides=["physics=newton_steel"])
    assert type(steel.sim.physics.solver_cfg) is not type(reference.sim.physics.solver_cfg)


def test_anymal_action_bounds_have_one_declaration():
    env, agent = resolve_task_config(TASKS[1], "rsl_rl_cfg_entry_point", overrides=["physics=newton_nerd"])
    recording, _ = resolve_task_config(
        TASKS[1], None, env_cfg_entry_point="nerd_recording_cfg_entry_point", overrides=[]
    )
    assert recording.actions.joint_targets.clip == env.actions.joint_pos.clip
    for low, high in env.actions.joint_pos.clip.values():
        assert agent.clip_actions * env.actions.joint_pos.scale == pytest.approx((high - low) / 2)


def test_optional_configuration_imports_without_neural_library():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
class Missing:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] == 'newton_neural':
            raise ModuleNotFoundError(name)
sys.meta_path.insert(0, Missing())
from isaaclab_tasks.utils import resolve_task_config
for task in ('Isaac-Reach-Franka', 'IsaacContrib-Velocity-Flat-AnymalC', 'Isaac-Reorient-Cube-Allegro'):
    resolve_task_config(
        task, 'nerd_cfg_entry_point', env_cfg_entry_point='nerd_recording_cfg_entry_point', overrides=[]
    )
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
