# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration ownership and the shared task-resolution boundary."""

import ast
import dataclasses
import subprocess
import sys

import pytest
from isaaclab_neural import NeRDCfg, NeRDTrainingCfg
from isaaclab_neural.record.recording_cfg import NeRDRecorderCfg

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


def test_osc_does_not_inherit_neural_physics():
    offered = enumerate_task_presets("Isaac-Reach-Franka-OSC")[PresetTarget.PHYSICS]
    assert "newton_nerd" not in offered
    assert "newton_nerd" in enumerate_task_presets(TASKS[0])[PresetTarget.PHYSICS]


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


def test_neural_ownership_boundaries():
    from pathlib import Path

    from isaaclab.paths import ISAACLAB_ROOT

    root = Path(ISAACLAB_ROOT)
    neural = root / "source/isaaclab_neural/isaaclab_neural"
    assert not (neural / "nerd_recipe.py").exists()
    assert not (neural / "nerd_recipe_cfg.py").exists()
    for name, owner in (("nerd_training_cfg.py", "rl"), ("recording.py", "record"), ("recording_cfg.py", "record")):
        assert not (neural / name).exists()
        assert (neural / owner / name).is_file()
    forbidden = {
        "RecordingSimulation",
        "EnvSimulation",
        "TrajectoryRecorder",
        "pop_initial_nerd",
        "forget_setup_changes",
        "record_nerd_step",
    }
    for path in neural.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                assert node.name not in forbidden, path
            if isinstance(node, ast.ImportFrom):
                assert not forbidden & {a.name for a in node.names}, path
            if path.parent.name == "physics" and isinstance(node, ast.Name):
                assert node.id != "SimulationContext", path
    assert not list((root / "source/isaaclab_tasks/isaaclab_tasks").rglob("nerd_recipe*.py"))
    for package in ("isaaclab", "isaaclab_newton"):
        for path in (root / "source" / package / package).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith("isaaclab_neural"), path
