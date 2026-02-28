# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Self-contained tests for Hydra configuration utilities.

These tests verify the REPLACE-only preset system without depending on
external environment configurations.
"""

import pytest

from isaaclab.utils import configclass

from isaaclab_tasks.utils.hydra import (
    apply_overrides,
    collect_presets,
    parse_overrides,
)
from isaaclab_tasks.utils.preset_cfg import PresetCfg

# =============================================================================
# Test Configuration Classes with Presets
# =============================================================================


@configclass
class JointPositionActionCfg:
    """Default joint position action config."""

    class_type: str = "JointPositionAction"
    asset_name: str = "robot"
    joint_names: list = [".*"]
    scale: float = 1.0
    offset: float = 0.0


@configclass
class RelativeJointPositionActionCfg:
    """Relative joint position action config."""

    class_type: str = "RelativeJointPositionAction"
    asset_name: str = "robot"
    joint_names: list = [".*"]
    scale: float = 0.2
    offset: float = 0.0
    use_zero_offset: bool = True


@configclass
class VelocityActionCfg:
    """Velocity action config."""

    class_type: str = "VelocityAction"
    asset_name: str = "robot"
    velocity_scale: float = 1.0


@configclass
class ArmActionCfgAutoDefault:
    """Arm action config using auto-default pattern (no inheritance)."""

    presets = {
        "default": JointPositionActionCfg(),
        "joint_position": JointPositionActionCfg(),
        "relative_joint_position": RelativeJointPositionActionCfg(),
    }


@configclass
class ArmActionCfgInheritance(JointPositionActionCfg):
    """Arm action config using inheritance pattern."""

    presets = {
        "joint_position": JointPositionActionCfg(),
        "relative_joint_position": RelativeJointPositionActionCfg(),
    }


@configclass
class PhysxCfg:
    """PhysX physics backend config."""

    backend: str = "physx"
    dt: float = 0.005
    substeps: int = 2


@configclass
class NewtonCfg:
    """Newton physics backend config."""

    backend: str = "newton"
    dt: float = 0.002
    substeps: int = 4
    solver_iterations: int = 8


@configclass
class SimBackendCfg(PresetCfg):
    """Physics backend presets using PresetCfg pattern."""

    default: PhysxCfg = PhysxCfg()
    newton: NewtonCfg = NewtonCfg()


@configclass
class PresetCfgSimCfg:
    """Sim config containing a PresetCfg-based backend field."""

    render_interval: int = 1
    backend: SimBackendCfg = SimBackendCfg()


@configclass
class PresetCfgEnvCfg:
    """Environment config for PresetCfg tests."""

    decimation: int = 4
    sim: PresetCfgSimCfg = PresetCfgSimCfg()


# Use auto-default pattern as the default for tests
ArmActionCfg = ArmActionCfgAutoDefault


@configclass
class JointControlActionsCfg:
    """Actions config with arm_action field."""

    arm_action: ArmActionCfg = ArmActionCfg()


@configclass
class VelocityControlActionsCfg:
    """Actions config with velocity_command (no arm_action)."""

    velocity_command: VelocityActionCfg = VelocityActionCfg()


@configclass
class NoiselessObservationsCfg:
    """Noiseless observations."""

    enable_corruption: bool = False
    concatenate_terms: bool = True
    noise_scale: float = 0.0


@configclass
class FastObservationsCfg:
    """Fast/inference observations - no corruption, simple."""

    enable_corruption: bool = False
    concatenate_terms: bool = False
    noise_scale: float = 0.0


@configclass
class ObservationsCfg:
    """Observations config with presets."""

    enable_corruption: bool = True
    concatenate_terms: bool = True
    noise_scale: float = 0.1

    presets = {
        "noise_less": NoiselessObservationsCfg(),
        "fast": FastObservationsCfg(),
    }


@configclass
class ActionsCfg:
    """Actions config with presets for different control modes."""

    arm_action: ArmActionCfg = ArmActionCfg()

    presets = {
        "joint_control": JointControlActionsCfg(),
        "velocity_control": VelocityControlActionsCfg(),
    }


@configclass
class SmallPolicyCfg:
    """Small policy network config."""

    actor_hidden_dims: list = [64, 32]


@configclass
class LargePolicyCfg:
    """Large policy network config."""

    actor_hidden_dims: list = [512, 256, 128]


@configclass
class FastPolicyCfg:
    """Fast/inference policy - smaller network for speed."""

    actor_hidden_dims: list = [32, 16]


@configclass
class PolicyCfg:
    """Policy config with presets."""

    actor_hidden_dims: list = [256, 128]

    presets = {
        "small_network": SmallPolicyCfg(),
        "large_network": LargePolicyCfg(),
        "fast": FastPolicyCfg(),
    }


@configclass
class SampleEnvCfg:
    """Sample environment config with nested configs."""

    decimation: int = 4
    sim_dt: float = 0.005

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()


@configclass
class SampleAgentCfg:
    """Sample agent config."""

    max_iterations: int = 1000
    learning_rate: float = 3e-4
    policy: PolicyCfg = PolicyCfg()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_configs():
    """Create fresh test configs and collect presets recursively."""
    env_cfg = SampleEnvCfg()
    agent_cfg = SampleAgentCfg()

    # Collect presets recursively (same as hydra.py does)
    presets = {
        "env": collect_presets(env_cfg),
        "agent": collect_presets(agent_cfg),
    }

    return env_cfg, agent_cfg, presets


# =============================================================================
# Tests for collect_presets
# =============================================================================


def test_collect_presets():
    """Test collecting presets from all levels of config tree."""
    env_cfg = SampleEnvCfg()
    presets = collect_presets(env_cfg)

    # Top-level presets
    assert "observations" in presets
    assert "actions" in presets
    assert "noise_less" in presets["observations"]
    assert "velocity_control" in presets["actions"]

    # Nested presets from actions.arm_action
    assert "actions.arm_action" in presets
    assert "relative_joint_position" in presets["actions.arm_action"]


# =============================================================================
# Tests for parse_overrides
# =============================================================================


def test_parse_overrides_mixed(test_configs):
    """Mix of all override types with proper categorization."""
    _, _, presets = test_configs
    args = [
        "presets=fast",  # global preset (applies to observations AND policy)
        "env.decimation=10",  # global scalar
        "env.observations=noise_less",  # path preset
        "env.actions.arm_action=relative_joint_position",  # nested preset
        "env.actions.arm_action.scale=2.0",  # preset scalar
    ]
    global_presets, preset_sel, preset_scalar, global_scalar = parse_overrides(args, presets)

    assert global_presets == ["fast"]
    assert ("env", "observations", "noise_less") in preset_sel
    assert ("env", "actions.arm_action", "relative_joint_position") in preset_sel
    assert ("env.actions.arm_action.scale", "2.0") in preset_scalar
    assert "env.decimation=10" in global_scalar


def test_parse_overrides_sorted_by_depth(test_configs):
    """Parent presets should be applied before children."""
    _, _, presets = test_configs
    args = [
        "env.actions.arm_action=relative_joint_position",  # nested (depth 2)
        "env.actions=joint_control",  # parent (depth 1)
    ]
    _, preset_sel, _, _ = parse_overrides(args, presets)

    # joint_control (depth 1) should come before arm_action (depth 2)
    assert preset_sel[0] == ("env", "actions", "joint_control")
    assert preset_sel[1] == ("env", "actions.arm_action", "relative_joint_position")


# =============================================================================
# Tests for apply_overrides
# =============================================================================


def test_apply_overrides_global_preset(test_configs):
    """Global preset should apply to all matching paths with same name."""
    env_cfg, agent_cfg, presets = test_configs
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}

    # Check defaults before applying
    assert env_cfg.observations.enable_corruption is True
    assert agent_cfg.policy.actor_hidden_dims == [256, 128]

    # Use global preset "fast" - should apply to BOTH observations AND policy
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, ["fast"], [], [], presets)

    # Should have applied FastObservationsCfg to env.observations
    assert env_cfg.observations.enable_corruption is False
    assert env_cfg.observations.concatenate_terms is False

    # Should have applied FastPolicyCfg to agent.policy
    assert agent_cfg.policy.actor_hidden_dims == [32, 16]


def test_apply_overrides_multiple_global_presets(test_configs):
    """Multiple non-conflicting global presets should all apply."""
    env_cfg, agent_cfg, presets = test_configs
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}

    # noise_less only affects observations, large_network only affects policy
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, ["noise_less", "large_network"], [], [], presets)

    # noise_less applied to observations
    assert env_cfg.observations.enable_corruption is False
    assert env_cfg.observations.noise_scale == 0.0

    # large_network applied to policy
    assert agent_cfg.policy.actor_hidden_dims == [512, 256, 128]


def test_apply_overrides_conflicting_global_presets(test_configs):
    """Conflicting global presets should raise ValueError."""
    env_cfg, agent_cfg, presets = test_configs
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}

    # Both "fast" and "noise_less" define a preset for observations -> conflict
    with pytest.raises(ValueError, match="Conflicting global presets"):
        apply_overrides(env_cfg, agent_cfg, hydra_cfg, ["fast", "noise_less"], [], [], presets)


def test_apply_overrides_preset_with_scalars(test_configs):
    """Preset selection + scalar overrides on new preset."""
    env_cfg, agent_cfg, presets = test_configs
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}

    preset_sel = [("env", "actions.arm_action", "relative_joint_position")]
    preset_scalar = [
        ("env.actions.arm_action.scale", "5.0"),
        ("env.actions.arm_action.use_zero_offset", "false"),
    ]

    apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], preset_sel, preset_scalar, presets)

    # Preset selected (RelativeJointPositionActionCfg has use_zero_offset)
    assert hasattr(env_cfg.actions.arm_action, "use_zero_offset")
    assert env_cfg.actions.arm_action.scale == 5.0
    assert env_cfg.actions.arm_action.use_zero_offset is False


def test_apply_overrides_nested_groups(test_configs):
    """Select parent group, then nested group, then scalar."""
    env_cfg, agent_cfg, presets = test_configs
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}

    preset_sel = [
        ("env", "actions", "joint_control"),
        ("env", "actions.arm_action", "relative_joint_position"),
    ]
    preset_scalar = [("env.actions.arm_action.scale", "7.0")]

    apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], preset_sel, preset_scalar, presets)

    assert isinstance(env_cfg.actions, JointControlActionsCfg)
    assert hasattr(env_cfg.actions.arm_action, "use_zero_offset")
    assert env_cfg.actions.arm_action.scale == 7.0


def test_apply_overrides_structural_replacement(test_configs):
    """Selecting velocity_control replaces structure (removes arm_action)."""
    env_cfg, agent_cfg, presets = test_configs
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}

    preset_sel = [("env", "actions", "velocity_control")]
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], preset_sel, [], presets)

    assert isinstance(env_cfg.actions, VelocityControlActionsCfg)
    assert hasattr(env_cfg.actions, "velocity_command")
    assert not hasattr(env_cfg.actions, "arm_action") or env_cfg.actions.arm_action is None


def test_apply_overrides_unknown_raises(test_configs):
    """Unknown preset group or name should raise ValueError."""
    env_cfg, agent_cfg, presets = test_configs
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}

    with pytest.raises(ValueError, match="Unknown preset group"):
        apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], [("env", "nonexistent", "x")], [], presets)

    with pytest.raises(ValueError, match="Unknown preset"):
        apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], [("env", "observations", "bad")], [], presets)

def test_preset_cfg_collect_presets():
    """Test that collect_presets discovers PresetCfg subclass fields as presets."""
    env_cfg = PresetCfgEnvCfg()
    presets = collect_presets(env_cfg)

    assert "sim.backend" in presets
    assert "default" in presets["sim.backend"]
    assert "newton" in presets["sim.backend"]
    assert isinstance(presets["sim.backend"]["default"], PhysxCfg)
    assert isinstance(presets["sim.backend"]["newton"], NewtonCfg)


def test_preset_cfg_auto_default():
    """Test that the 'default' field is auto-applied when no CLI override is given."""
    env_cfg = PresetCfgEnvCfg()
    presets = {
        "env": collect_presets(env_cfg),
        "agent": {},
    }
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": {}}

    apply_overrides(env_cfg, None, hydra_cfg, [], [], [], presets)

    assert env_cfg.sim.backend.backend == "physx"
    assert env_cfg.sim.backend.dt == 0.005


def test_preset_cfg_cli_selection():
    """Test that CLI selection replaces with the chosen preset."""
    env_cfg = PresetCfgEnvCfg()
    presets = {
        "env": collect_presets(env_cfg),
        "agent": {},
    }
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": {}}

    preset_sel = [("env", "sim.backend", "newton")]
    apply_overrides(env_cfg, None, hydra_cfg, [], preset_sel, [], presets)

    assert isinstance(env_cfg.sim.backend, NewtonCfg)
    assert env_cfg.sim.backend.backend == "newton"
    assert env_cfg.sim.backend.dt == 0.002
    assert env_cfg.sim.backend.solver_iterations == 8


def test_preset_cfg_global_preset():
    """Test that a global preset applies to PresetCfg-discovered presets."""
    env_cfg = PresetCfgEnvCfg()
    presets = {
        "env": collect_presets(env_cfg),
        "agent": {},
    }
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": {}}

    apply_overrides(env_cfg, None, hydra_cfg, ["newton"], [], [], presets)

    assert isinstance(env_cfg.sim.backend, NewtonCfg)
    assert env_cfg.sim.backend.backend == "newton"


def test_preset_cfg_with_presets_attr_raises():
    """PresetCfg subclass with a 'presets' attribute should raise ValueError."""

    @configclass
    class BadBackendCfg(PresetCfg):
        default: PhysxCfg = PhysxCfg()
        newton: NewtonCfg = NewtonCfg()
        presets = {"extra": PhysxCfg()}

    @configclass
    class BadSimCfg:
        backend: BadBackendCfg = BadBackendCfg()

    @configclass
    class BadEnvCfg:
        sim: BadSimCfg = BadSimCfg()

    with pytest.raises(ValueError, match="must not define a 'presets' attribute"):
        collect_presets(BadEnvCfg())
