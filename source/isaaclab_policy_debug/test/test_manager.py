# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from types import SimpleNamespace

import torch
from isaaclab_policy_debug.catalog import CheckpointEntry, LoadedCheckpoint
from isaaclab_policy_debug.config import PolicyDebugCfg
from isaaclab_policy_debug.manager import PolicyDebugManager


class _Visualizer:
    def __init__(self):
        self.configured = []
        self.visible = []
        self.layouts = []
        self.styles = []
        self.shape_visibility = []
        self.asset_requests = []
        self.frames = 0

    def scene_asset_shape_visibility(self, asset_names):
        request = tuple(asset_names)
        self.asset_requests.append(request)
        if request == ("robot", "cube", "table", "target_pad"):
            return (True, True, False)
        return (True, False, True)

    def configure_environment_layers(self, ids):
        self.configured.append(list(ids))

    def set_visible_environment_ids(self, ids):
        self.visible.append(list(ids))

    def set_environment_shape_visibility(self, visibility):
        self.shape_visibility.append(visibility)

    def set_environment_layout(self, layout):
        self.layouts.append(layout)

    def set_environment_render_styles(self, styles):
        self.styles.append(styles)

    def frame_visible_environments(self):
        self.frames += 1

    def is_running(self):
        return False


class _Policy:
    def __init__(self, value):
        self.value = value
        self.resets = 0
        self.observations = []

    def __call__(self, obs):
        self.observations.append(obs)
        return torch.full((1, 1), self.value)

    def reset(self):
        self.resets += 1


class _ParameterPolicy(_Policy):
    def __init__(self, value):
        super().__init__(value)
        self.parameter = torch.nn.Parameter(torch.tensor(float(value)))

    def __call__(self, obs):
        return self.parameter.reshape(1, 1)


class _Factory:
    def create(self, checkpoint):
        return _Policy(float(checkpoint.iteration))


class _ParameterFactory:
    def create(self, checkpoint):
        return _ParameterPolicy(float(checkpoint.iteration))


class _Loader:
    def load(self, entry):
        entry.iteration = entry.filename_iteration
        return LoadedCheckpoint(entry.path, {}, entry.iteration, {}, {})


class _Scenario:
    def __init__(self):
        self.resets = []
        self.failures = {}
        self.before_steps = []
        self.after_steps = []
        self.automatic_reset = False

    def comparison_visible_assets(self):
        return ("robot", "cube", "table", "target_pad")

    def overlay_visible_assets(self):
        return ("robot", "cube")

    def validate_checkpoint(self, checkpoint, env):
        return None

    def reset_synchronized(self, env, ids):
        self.resets.append(list(ids))

    def rollout_failures(self, env, ids):
        return self.failures

    def before_step(self, env, ids):
        self.before_steps.append(list(ids))

    def after_step(self, env, ids):
        self.after_steps.append(list(ids))

    def accept_automatic_reset(self, env, ids):
        return self.automatic_reset


class _Env:
    def __init__(self, capacity):
        self.observations = {"policy": torch.arange(capacity, dtype=torch.float32).unsqueeze(1)}
        self.unwrapped = SimpleNamespace(device="cpu", sim=SimpleNamespace(render=lambda: None))
        self.last_actions = None
        self.done_slot = None

    def get_observations(self):
        return self.observations

    def step(self, actions):
        self.last_actions = actions.clone()
        dones = torch.zeros(len(actions), dtype=torch.bool)
        if self.done_slot is not None:
            dones[self.done_slot] = True
            self.done_slot = None
        return self.observations, None, dones, {}


def _entry(path: Path, iteration: int) -> CheckpointEntry:
    path.write_bytes(b"x")
    return CheckpointEntry(path, 1, path.stat().st_mtime_ns, stable_scans=2, ready=True, iteration=iteration)


def test_manager_activates_manually_routes_rows_and_zeroes_inactive_actions(tmp_path: Path):
    cfg = PolicyDebugCfg(tmp_path, max_policies=3)
    env = _Env(3)
    viz = _Visualizer()
    scenario = _Scenario()
    manager = PolicyDebugManager(cfg, env, viz, scenario, _Factory(), action_dim=1)
    manager.loader = _Loader()
    first = _entry(tmp_path / "model_2.pt", 2)
    second = _entry(tmp_path / "model_8.pt", 8)

    manager.set_checkpoint_enabled(first, True)
    manager.set_checkpoint_enabled(second, True)
    manager.step()
    assert viz.configured == [[0, 1, 2]]
    assert manager.active_slots == [0, 1]
    assert scenario.resets == [[0, 1]]
    assert env.last_actions[:, 0].tolist() == [2.0, 8.0, 0.0]
    assert all(active.policy.resets == 1 for active in manager.active.values())
    assert manager.active[first.path].policy.observations[0]["policy"].tolist() == [[0.0]]
    assert manager.active[second.path].policy.observations[0]["policy"].tolist() == [[1.0]]

    manager.set_checkpoint_enabled(first, False)
    assert manager.active_slots == [1]
    third = _entry(tmp_path / "model_9.pt", 9)
    manager.set_checkpoint_enabled(third, True)
    manager.apply_frame_boundary_requests()
    assert manager.active[third.path].slot == 0


def test_overlay_reference_reassigns_and_any_active_done_restarts_group(tmp_path: Path):
    cfg = PolicyDebugCfg(tmp_path, max_policies=3)
    env = _Env(3)
    viz = _Visualizer()
    scenario = _Scenario()
    manager = PolicyDebugManager(cfg, env, viz, scenario, _Factory(), action_dim=1)
    manager.loader = _Loader()
    older = _entry(tmp_path / "model_2.pt", 2)
    latest = _entry(tmp_path / "model_8.pt", 8)
    manager.set_checkpoint_enabled(older, True)
    manager.set_checkpoint_enabled(latest, True)
    manager.apply_frame_boundary_requests()

    manager.set_overlay(True)
    styles = viz.styles[-1]
    visibility = viz.shape_visibility[-1]
    older_style = styles[manager.active[older.path].slot]
    assert older_style.opacity == 0.25
    assert older_style.color == manager.active[older.path].tint
    assert manager.ghost_tint(manager.active[older.path]) == manager.active[older.path].tint
    assert visibility[manager.active[older.path].slot] == (True, False, False)
    latest_style = styles[manager.active[latest.path].slot]
    assert latest_style.opacity == 1.0
    assert latest_style.color is None
    assert manager.reference_policy is manager.active[latest.path]
    assert manager.ghost_tint(manager.active[latest.path]) is None
    assert visibility[manager.active[latest.path].slot] == (True, True, False)
    assert viz.asset_requests == [("robot", "cube", "table", "target_pad"), ("robot", "cube")]

    env.done_slot = manager.active[older.path].slot
    manager.step()
    assert scenario.resets == [[0, 1], [0, 1]]

    manager.set_checkpoint_enabled(latest, False)
    styles = viz.styles[-1]
    assert viz.shape_visibility[-1][manager.active[older.path].slot] == (True, True, False)
    assert styles[manager.active[older.path].slot].opacity == 1.0
    assert manager.reference_policy is manager.active[older.path]
    assert manager.ghost_tint(manager.active[older.path]) is None


def test_verified_same_step_autoreset_does_not_reset_physics_twice(tmp_path: Path):
    cfg = PolicyDebugCfg(tmp_path, max_policies=2)
    env = _Env(2)
    scenario = _Scenario()
    scenario.automatic_reset = True
    manager = PolicyDebugManager(cfg, env, _Visualizer(), scenario, _Factory(), action_dim=1)
    manager.loader = _Loader()
    first = _entry(tmp_path / "model_2.pt", 2)
    second = _entry(tmp_path / "model_8.pt", 8)
    manager.set_checkpoint_enabled(first, True)
    manager.set_checkpoint_enabled(second, True)
    manager.apply_frame_boundary_requests()

    env.done_slot = manager.active[first.path].slot
    manager.step()

    assert scenario.resets == [[0, 1]]
    assert scenario.before_steps == [[0, 1]]
    assert scenario.after_steps == [[0, 1]]
    assert all(active.policy.resets == 2 for active in manager.active.values())


def test_numerical_failure_deactivates_only_its_policy_slot(tmp_path: Path):
    cfg = PolicyDebugCfg(tmp_path, max_policies=3)
    env = _Env(3)
    scenario = _Scenario()
    manager = PolicyDebugManager(cfg, env, _Visualizer(), scenario, _Factory(), action_dim=1)
    manager.loader = _Loader()
    stable = _entry(tmp_path / "model_8.pt", 8)
    unstable = _entry(tmp_path / "model_2.pt", 2)
    manager.set_checkpoint_enabled(stable, True)
    manager.set_checkpoint_enabled(unstable, True)
    manager.apply_frame_boundary_requests()
    unstable_slot = manager.active[unstable.path].slot
    scenario.failures = {unstable_slot: "cube state became non-finite"}

    manager.step()

    assert manager.active_slots == [manager.active[stable.path].slot]
    assert unstable.status == "error"
    assert "Numerically unstable rollout" in unstable.error
    assert scenario.resets == [[0, 1], [manager.active[stable.path].slot]]


def test_manager_detaches_policy_actions_before_environment_step(tmp_path: Path):
    cfg = PolicyDebugCfg(tmp_path, max_policies=1)
    env = _Env(1)
    manager = PolicyDebugManager(cfg, env, _Visualizer(), _Scenario(), _ParameterFactory(), action_dim=1)
    manager.loader = _Loader()
    entry = _entry(tmp_path / "model_4.pt", 4)
    manager.set_checkpoint_enabled(entry, True)

    manager.step()

    assert env.last_actions is not None
    assert env.last_actions.requires_grad is False
