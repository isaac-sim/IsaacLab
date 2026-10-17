# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-component variant composition, cached resets, and policy contracts."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from isaaclab.managers import EventTermCfg
from isaaclab.utils import math as math_utils

from isaaclab_tasks.contrib.nist.assembly_profile import AssemblyProfile
from isaaclab_tasks.contrib.nist.assembly_variants import ASSEMBLY_VARIANTS
from isaaclab_tasks.contrib.nist.config.agents.models import MLPEncoder, SimBaModel
from isaaclab_tasks.contrib.nist.factory_env_cfg import FactoryEnvCfg
from isaaclab_tasks.contrib.nist.factory_variant_env_cfg import FactoryVariantEnvCfg
from isaaclab_tasks.contrib.nist.utils import reset_state
from isaaclab_tasks.contrib.nist.utils.sampling import SamplerCfg
from isaaclab_tasks.contrib.nist.utils.variant_event_combinators import variant_reset_accumulator
from isaaclab_tasks.utils.hydra import resolve_task_config
from isaaclab_tasks.utils.success_monitor import SuccessMonitorCfg


def test_static_and_variant_tasks_resolve_with_their_policy_inputs():
    """Both public task IDs remain usable, and play mode selects random starts."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, gymnasium, isaaclab_tasks; "
            "assert 'IsaacContrib-Factory-Variant-Franka' in gymnasium.registry; "
            "assert 'isaaclab_tasks.contrib.nist.factory_presets' not in sys.modules",
        ],
        check=True,
    )
    static, static_agent = resolve_task_config("IsaacContrib-Factory-Franka", "rsl_rl_cfg_entry_point", overrides=[])
    variant, variant_agent = resolve_task_config(
        "IsaacContrib-Factory-Variant-Franka", "rsl_rl_cfg_entry_point", play_mode=True, overrides=[]
    )
    assert type(static) is FactoryEnvCfg
    assert isinstance(variant, FactoryVariantEnvCfg)
    assert len(variant.events.assembly_variants.params["variant_names"]) == 20
    assert len(variant.scene.fixed_asset.spawn.assets_cfg) == len(variant.scene.held_asset.spawn.assets_cfg) == 20
    assert static_agent.obs_groups == {"actor": ["policy"], "critic": ["policy"]}
    assert variant_agent.obs_groups == {"actor": ["policy", "perception"], "critic": ["policy", "perception"]}
    for group in variant_agent.obs_groups["actor"]:
        assert getattr(variant.observations, group) is not None
    reset = variant.events.reset_strategies.params["reset_term"]
    choice = reset.params["terms"]["reset_strategies"]
    assert list(choice.params["terms"]) == ["start_random"]
    assert variant.episode_length_s == 14.0


def test_simba_point_cloud_encoder_trains_and_exports():
    """Point clouds contribute gradients, and exported inference matches the actor."""
    observations = TensorDict({"policy": torch.randn(4, 7), "perception": torch.randn(4, 45)}, batch_size=[4])
    model = SimBaModel(
        observations,
        {"actor": ["policy", "perception"]},
        "actor",
        output_dim=2,
        hidden_dim=16,
        obs_normalization=True,
        encoder_cfg={"perception": {"class_name": MLPEncoder, "hidden_dims": [16], "output_dim": 6}},
    )
    model.update_normalization(observations)
    output = model(observations)
    assert output.shape == (4, 2)
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    exported = torch.jit.script(model.as_jit())
    torch.testing.assert_close(exported(observations["policy"], observations["perception"]), output)


def test_cached_reset_restores_variant_before_pose_and_preserves_input():
    """Cached variant identity precedes pose restoration, with environment offsets applied once."""
    calls = []

    class Asset:
        num_mesh_variants = 20
        root_view = SimpleNamespace(is_fixed_base=False)
        mesh_variant_ids = SimpleNamespace(torch=torch.tensor([2, 7], dtype=torch.int32))
        data = SimpleNamespace(root_state_w=SimpleNamespace(torch=torch.arange(26).float().reshape(2, 13)))

        def write_mesh_variant_to_sim(self, variant_ids, env_ids):
            calls.append(("mesh", variant_ids.clone(), env_ids.clone()))

        def write_root_link_pose_to_sim_index(self, *, root_pose, env_ids):
            calls.append(("pose", root_pose.clone(), env_ids.clone()))

        def write_root_com_velocity_to_sim_index(self, *, root_velocity, env_ids):
            calls.append(("velocity", root_velocity.clone(), env_ids.clone()))

    asset = Asset()
    origins = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    env = SimpleNamespace(scene=SimpleNamespace(_articulations={}, _rigid_objects={"part": asset}, env_origins=origins))
    ids = torch.tensor([1, 0])
    packed = reset_state.get_reset_state(env, ids, ["part"], is_relative=True)
    torch.testing.assert_close(packed[:, 0], torch.tensor([7.0, 2.0]))
    torch.testing.assert_close(packed[:, 1:4], asset.data.root_state_w.torch[ids, :3] - origins[ids])
    original = packed.clone()
    reset_state.set_reset_state(env, packed, ids, ["part"], is_relative=True)
    assert [call[0] for call in calls] == ["mesh", "pose", "velocity"]
    torch.testing.assert_close(calls[0][1], torch.tensor([7, 2], dtype=torch.int32))
    torch.testing.assert_close(calls[1][1], asset.data.root_state_w.torch[ids, :7])
    torch.testing.assert_close(calls[2][1], asset.data.root_state_w.torch[ids, 7:])
    torch.testing.assert_close(packed, original)


@pytest.mark.parametrize("name", ["gear_mesh_small", "gear_mesh_medium", "gear_mesh_large"])
def test_gear_profile_reaches_assembled_and_entry_poses(name):
    variant = next(variant for variant in ASSEMBLY_VARIANTS if variant.name == name)
    segment = variant.profile.segments[0]
    profile = AssemblyProfile(variant.profile)
    assembled, _ = profile.sample((0.0, 0.0), 1, "cpu")
    entry, _ = profile.sample((1.0, 1.0), 1, "cpu")
    torch.testing.assert_close(assembled[0], torch.tensor(segment.start_pose.pos))
    torch.testing.assert_close(entry[0], torch.tensor(segment.start_pose.pos) + torch.tensor(segment.distance))


@pytest.mark.parametrize("fraction", [0.2, 0.4, 0.7])
def test_bnc_profile_turns_then_withdraws(fraction):
    """The bayonet segment turns through a quarter circle before linear withdrawal."""
    variant = next(variant for variant in ASSEMBLY_VARIANTS if variant.name == "bnc")
    twist, withdrawal = variant.profile.segments
    position, orientation = AssemblyProfile(variant.profile).sample((fraction, fraction), 1, "cpu")
    segment = twist if fraction <= twist.fraction[1] else withdrawal
    progress = (fraction - segment.fraction[0]) / (segment.fraction[1] - segment.fraction[0])
    expected_position = torch.lerp(torch.tensor(segment.start_pose.pos), torch.tensor(segment.end_pose.pos), progress)
    angle = torch.tensor([min(fraction / twist.fraction[1], 1.0) * torch.pi / 2])
    expected_orientation = math_utils.quat_mul(
        torch.tensor([twist.start_pose.quat]),
        math_utils.quat_from_euler_xyz(torch.zeros_like(angle), torch.zeros_like(angle), angle),
    )
    torch.testing.assert_close(position[0], expected_position, atol=1.0e-7, rtol=0)
    assert math_utils.quat_error_magnitude(orientation, expected_orientation).item() < 1.0e-4


@pytest.mark.parametrize("env_ids", [slice(None), slice(1, 4, 2), slice(3, 1), torch.tensor([3, 0])])
def test_reset_selectors_preserve_unselected_states_and_monitor_only_completed_attempts(env_ids, monkeypatch):
    """Partial or empty resets must not overwrite other environments or record nonexistent attempts."""
    progress = SimpleNamespace(is_success=torch.tensor([True, False, True, False]))
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        scene=SimpleNamespace(_articulations={}, _rigid_objects={}),
        extras={},
        termination_manager=SimpleNamespace(
            get_term_cfg=lambda _: SimpleNamespace(func=progress),
            _term_names=["abnormal"],
            _term_name_to_term_idx={"abnormal": 0},
            _last_episode_dones=torch.tensor([[False], [False], [True], [False]]),
        ),
    )
    cfg = EventTermCfg(
        func=variant_reset_accumulator,
        mode="reset",
        params={
            "reset_assets": [],
            "acceptance_conditions": {},
            "state_table_size": 4,
            "success_monitor_cfg": SuccessMonitorCfg(),
            "sampling": SamplerCfg(),
            "monitor_exclude_terms": ["abnormal"],
        },
    )
    accumulator = variant_reset_accumulator(cfg, env)
    accumulator.precollecting_phase = False
    accumulator.sampled_slots[:] = torch.tensor([3, -1, 2, 1])
    accumulator.state_cell_indices[:] = torch.arange(4)
    accumulator._sampler = object()
    updates, restores = [], []
    accumulator.success_monitor = SimpleNamespace(
        success_update=lambda slots, successes: updates.append((slots, successes))
    )
    monkeypatch.setattr(accumulator, "_sample_marginally_balanced", lambda n: (torch.ones(4) / 4, torch.arange(n)))
    monkeypatch.setattr(reset_state, "set_reset_state", lambda env, states, ids, *args, **kwargs: restores.append(ids))
    before = accumulator.sampled_slots.clone()
    selected = torch.arange(4)[env_ids]
    eligible = selected[(before[selected] >= 0) & ~env.termination_manager._last_episode_dones[selected, 0]]
    accumulator(env, env_ids, **cfg.params)
    expected = before.clone()
    expected[selected] = torch.arange(len(selected))
    torch.testing.assert_close(accumulator.sampled_slots, expected)
    assert len(restores) == int(bool(len(selected)))
    if len(selected):
        torch.testing.assert_close(restores[0], selected)
    assert len(updates) == int(bool(len(eligible)))
    if len(eligible):
        torch.testing.assert_close(updates[0][0], before[eligible])
        torch.testing.assert_close(updates[0][1], progress.is_success[eligible])
