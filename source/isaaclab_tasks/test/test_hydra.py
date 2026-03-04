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
    PresetCfg,
    apply_overrides,
    collect_presets,
    parse_overrides,
    resolve_preset_defaults,
)

# =============================================================================
# Leaf config classes (reused across all test sections)
# =============================================================================


@configclass
class JointPositionActionCfg:
    class_type: str = "JointPositionAction"
    asset_name: str = "robot"
    joint_names: list = [".*"]
    scale: float = 1.0
    offset: float = 0.0


@configclass
class RelativeJointPositionActionCfg:
    class_type: str = "RelativeJointPositionAction"
    asset_name: str = "robot"
    joint_names: list = [".*"]
    scale: float = 0.2
    offset: float = 0.0
    use_zero_offset: bool = True


@configclass
class VelocityActionCfg:
    class_type: str = "VelocityAction"
    asset_name: str = "robot"
    velocity_scale: float = 1.0


@configclass
class PhysxCfg:
    backend: str = "physx"
    dt: float = 0.005
    substeps: int = 2


@configclass
class NewtonCfg:
    backend: str = "newton"
    dt: float = 0.002
    substeps: int = 4
    solver_iterations: int = 8


@configclass
class NoiselessObservationsCfg:
    enable_corruption: bool = False
    concatenate_terms: bool = True
    noise_scale: float = 0.0


@configclass
class FastObservationsCfg:
    enable_corruption: bool = False
    concatenate_terms: bool = False
    noise_scale: float = 0.0


@configclass
class SmallPolicyCfg:
    actor_hidden_dims: list = [64, 32]


@configclass
class LargePolicyCfg:
    actor_hidden_dims: list = [512, 256, 128]


@configclass
class FastPolicyCfg:
    actor_hidden_dims: list = [32, 16]


# =============================================================================
# Composite configs using presets dict (Style 1-3)
# =============================================================================


@configclass
class ArmActionCfg:
    presets = {
        "default": JointPositionActionCfg(),
        "joint_position": JointPositionActionCfg(),
        "relative_joint_position": RelativeJointPositionActionCfg(),
    }


@configclass
class JointControlActionsCfg:
    arm_action: ArmActionCfg = ArmActionCfg()


@configclass
class VelocityControlActionsCfg:
    velocity_command: VelocityActionCfg = VelocityActionCfg()


@configclass
class ObservationsCfg:
    enable_corruption: bool = True
    concatenate_terms: bool = True
    noise_scale: float = 0.1
    presets = {
        "noise_less": NoiselessObservationsCfg(),
        "fast": FastObservationsCfg(),
    }


@configclass
class ActionsCfg:
    arm_action: ArmActionCfg = ArmActionCfg()
    presets = {
        "joint_control": JointControlActionsCfg(),
        "velocity_control": VelocityControlActionsCfg(),
    }


@configclass
class PolicyCfg:
    actor_hidden_dims: list = [256, 128]
    presets = {
        "small_network": SmallPolicyCfg(),
        "large_network": LargePolicyCfg(),
        "fast": FastPolicyCfg(),
    }


@configclass
class SampleEnvCfg:
    decimation: int = 4
    sim_dt: float = 0.005
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()


@configclass
class SampleAgentCfg:
    max_iterations: int = 1000
    learning_rate: float = 3e-4
    policy: PolicyCfg = PolicyCfg()


# =============================================================================
# Composite configs using PresetCfg (Style 4)
# =============================================================================


@configclass
class SimBackendCfg(PresetCfg):
    default: PhysxCfg = PhysxCfg()
    newton: NewtonCfg = NewtonCfg()


@configclass
class ObsModeCfg(PresetCfg):
    default: NoiselessObservationsCfg = NoiselessObservationsCfg()
    fast: FastObservationsCfg = FastObservationsCfg()


@configclass
class PolicyModeCfg(PresetCfg):
    default: SmallPolicyCfg = SmallPolicyCfg()
    fast: FastPolicyCfg = FastPolicyCfg()


@configclass
class PresetCfgEnvCfg:
    decimation: int = 4
    backend: SimBackendCfg = SimBackendCfg()
    observations: ObsModeCfg = ObsModeCfg()


@configclass
class PresetCfgAgentCfg:
    learning_rate: float = 3e-4
    policy: PolicyModeCfg = PolicyModeCfg()


@configclass
class RootAgentCfg(PresetCfg):
    """Root-level PresetCfg — the agent config itself is a PresetCfg."""

    default: SampleAgentCfg = SampleAgentCfg()
    fast: SampleAgentCfg = SampleAgentCfg(max_iterations=100, learning_rate=1e-3)


# -- Nested PresetCfg-inside-PresetCfg (mirrors scene.base_camera pattern) --


@configclass
class CameraSmallCfg:
    width: int = 64
    height: int = 64


@configclass
class CameraLargeCfg:
    width: int = 256
    height: int = 256


@configclass
class CameraPresetCfg(PresetCfg):
    small: CameraSmallCfg = CameraSmallCfg()
    large: CameraLargeCfg = CameraLargeCfg()
    default: CameraSmallCfg = CameraSmallCfg()


@configclass
class BaseSceneCfg:
    num_envs: int = 1024
    camera: CameraPresetCfg | None = None


@configclass
class ScenePresetCfg(PresetCfg):
    default: BaseSceneCfg = BaseSceneCfg()
    with_camera: BaseSceneCfg = BaseSceneCfg(camera=CameraPresetCfg())


@configclass
class NestedPresetEnvCfg:
    decimation: int = 4
    scene: ScenePresetCfg = ScenePresetCfg()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def dict_presets():
    """Fresh configs using presets dict pattern."""
    env_cfg = SampleEnvCfg()
    agent_cfg = SampleAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    return env_cfg, agent_cfg, presets


@pytest.fixture
def class_presets():
    """Fresh configs using PresetCfg pattern."""
    env_cfg = PresetCfgEnvCfg()
    agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    return env_cfg, agent_cfg, presets


# =============================================================================
# Tests: collect_presets
# =============================================================================


def test_collect_presets_dict_style():
    """presets dict discovered at correct paths."""
    presets = collect_presets(SampleEnvCfg())
    assert "observations" in presets
    assert "actions" in presets
    assert "actions.arm_action" in presets
    assert "noise_less" in presets["observations"]
    assert "velocity_control" in presets["actions"]
    assert "relative_joint_position" in presets["actions.arm_action"]


def test_collect_presets_class_style():
    """PresetCfg fields discovered at correct paths."""
    presets = collect_presets(PresetCfgEnvCfg())
    assert "backend" in presets
    assert set(presets["backend"].keys()) == {"default", "newton"}
    assert isinstance(presets["backend"]["default"], PhysxCfg)
    assert isinstance(presets["backend"]["newton"], NewtonCfg)


def test_collect_presets_root_level():
    """Root-level PresetCfg collected at path=''."""
    presets = collect_presets(RootAgentCfg())
    assert "" in presets
    assert set(presets[""].keys()) == {"default", "fast"}
    assert isinstance(presets[""]["default"], SampleAgentCfg)
    assert presets[""]["fast"].max_iterations == 100


def test_collect_presets_class_with_presets_attr_raises():
    """PresetCfg subclass with a 'presets' attribute raises ValueError."""

    @configclass
    class BadCfg(PresetCfg):
        default: PhysxCfg = PhysxCfg()
        presets = {"extra": PhysxCfg()}

    @configclass
    class WrapperCfg:
        child: BadCfg = BadCfg()

    with pytest.raises(ValueError, match="must not define a 'presets' attribute"):
        collect_presets(WrapperCfg())


# =============================================================================
# Tests: parse_overrides
# =============================================================================


def test_parse_overrides_mixed(dict_presets):
    """All override types categorized correctly."""
    _, _, presets = dict_presets
    args = [
        "presets=fast",
        "env.decimation=10",
        "env.observations=noise_less",
        "env.actions.arm_action=relative_joint_position",
        "env.actions.arm_action.scale=2.0",
    ]
    global_p, sel, scalar, glob = parse_overrides(args, presets)
    assert global_p == ["fast"]
    assert ("env", "observations", "noise_less") in sel
    assert ("env", "actions.arm_action", "relative_joint_position") in sel
    assert ("env.actions.arm_action.scale", "2.0") in scalar
    assert "env.decimation=10" in glob


def test_parse_overrides_sorted_by_depth(dict_presets):
    """Parent presets applied before children."""
    _, _, presets = dict_presets
    args = ["env.actions.arm_action=relative_joint_position", "env.actions=joint_control"]
    _, sel, _, _ = parse_overrides(args, presets)
    assert sel[0] == ("env", "actions", "joint_control")
    assert sel[1] == ("env", "actions.arm_action", "relative_joint_position")


def test_parse_overrides_root_preset():
    """Root-level PresetCfg parsed as agent=<name>."""
    presets = {"env": {}, "agent": collect_presets(RootAgentCfg())}
    _, sel, _, _ = parse_overrides(["agent=fast"], presets)
    assert sel == [("agent", "", "fast")]


# =============================================================================
# Tests: apply_overrides — presets dict
# =============================================================================


def test_apply_global_preset(dict_presets):
    """Global preset applies to all matching paths."""
    env_cfg, agent_cfg, presets = dict_presets
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, ["fast"], [], [], presets)
    assert env_cfg.observations.enable_corruption is False
    assert agent_cfg.policy.actor_hidden_dims == [32, 16]


def test_apply_multiple_global_presets(dict_presets):
    """Non-conflicting global presets all apply."""
    env_cfg, agent_cfg, presets = dict_presets
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, ["noise_less", "large_network"], [], [], presets)
    assert env_cfg.observations.noise_scale == 0.0
    assert agent_cfg.policy.actor_hidden_dims == [512, 256, 128]


def test_apply_conflicting_global_raises(dict_presets):
    """Conflicting global presets raise ValueError."""
    env_cfg, agent_cfg, presets = dict_presets
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    with pytest.raises(ValueError, match="Conflicting global presets"):
        apply_overrides(env_cfg, agent_cfg, hydra_cfg, ["fast", "noise_less"], [], [], presets)


def test_apply_preset_with_scalars(dict_presets):
    """Preset selection + scalar overrides."""
    env_cfg, agent_cfg, presets = dict_presets
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    sel = [("env", "actions.arm_action", "relative_joint_position")]
    scalars = [("env.actions.arm_action.scale", "5.0"), ("env.actions.arm_action.use_zero_offset", "false")]
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], sel, scalars, presets)
    assert env_cfg.actions.arm_action.scale == 5.0
    assert env_cfg.actions.arm_action.use_zero_offset is False


def test_apply_nested_groups(dict_presets):
    """Select parent group, then nested group, then scalar."""
    env_cfg, agent_cfg, presets = dict_presets
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    sel = [("env", "actions", "joint_control"), ("env", "actions.arm_action", "relative_joint_position")]
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], sel, [("env.actions.arm_action.scale", "7.0")], presets)
    assert isinstance(env_cfg.actions, JointControlActionsCfg)
    assert env_cfg.actions.arm_action.scale == 7.0


def test_apply_structural_replacement(dict_presets):
    """Selecting velocity_control replaces structure."""
    env_cfg, agent_cfg, presets = dict_presets
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], [("env", "actions", "velocity_control")], [], presets)
    assert isinstance(env_cfg.actions, VelocityControlActionsCfg)
    assert hasattr(env_cfg.actions, "velocity_command")


def test_apply_unknown_raises(dict_presets):
    """Unknown preset group or name raises ValueError."""
    env_cfg, agent_cfg, presets = dict_presets
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    with pytest.raises(ValueError, match="Unknown preset group"):
        apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], [("env", "nonexistent", "x")], [], presets)
    with pytest.raises(ValueError, match="Unknown preset"):
        apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], [("env", "observations", "bad")], [], presets)


# =============================================================================
# Tests: apply_overrides — PresetCfg (nested + broadcast + root)
# =============================================================================


def test_presetcfg_auto_default(class_presets):
    """'default' field auto-applied when no CLI override."""
    env_cfg, agent_cfg, presets = class_presets
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], [], [], presets)
    assert isinstance(env_cfg.backend, PhysxCfg)
    assert isinstance(env_cfg.observations, NoiselessObservationsCfg)
    assert isinstance(agent_cfg.policy, SmallPolicyCfg)


def test_presetcfg_cli_selection(class_presets):
    """Path selection replaces with chosen preset."""
    env_cfg, agent_cfg, presets = class_presets
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], [("env", "backend", "newton")], [], presets)
    assert isinstance(env_cfg.backend, NewtonCfg)
    assert env_cfg.backend.dt == 0.002


def test_presetcfg_global_broadcast(class_presets):
    """Global preset 'fast' broadcasts across env and agent PresetCfg fields."""
    env_cfg, agent_cfg, presets = class_presets
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, ["fast"], [], [], presets)
    assert isinstance(env_cfg.observations, FastObservationsCfg)
    assert isinstance(agent_cfg.policy, FastPolicyCfg)


def test_presetcfg_path_selection_others_default(class_presets):
    """Path preset on one field, others get auto-default."""
    env_cfg, agent_cfg, presets = class_presets
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], [("env", "backend", "newton")], [], presets)
    assert isinstance(env_cfg.backend, NewtonCfg)
    assert isinstance(env_cfg.observations, NoiselessObservationsCfg)
    assert isinstance(agent_cfg.policy, SmallPolicyCfg)


def test_root_presetcfg_auto_default():
    """Root-level PresetCfg auto-applies 'default'."""
    agent_cfg = RootAgentCfg()
    env_cfg = SampleEnvCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    env_cfg, agent_cfg = apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], [], [], presets)
    assert isinstance(agent_cfg, SampleAgentCfg)
    assert agent_cfg.max_iterations == 1000


def test_root_presetcfg_cli_selection():
    """Root-level PresetCfg resolved via path selection."""
    agent_cfg = RootAgentCfg()
    env_cfg = SampleEnvCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    env_cfg, agent_cfg = apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], [("agent", "", "fast")], [], presets)
    assert isinstance(agent_cfg, SampleAgentCfg)
    assert agent_cfg.max_iterations == 100
    assert agent_cfg.learning_rate == 1e-3


def test_root_presetcfg_global_preset():
    """Root-level PresetCfg resolved via global preset."""
    agent_cfg = RootAgentCfg()
    env_cfg = SampleEnvCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    env_cfg, agent_cfg = apply_overrides(env_cfg, agent_cfg, hydra_cfg, ["fast"], [], [], presets)
    assert isinstance(agent_cfg, SampleAgentCfg)
    assert agent_cfg.max_iterations == 100


# =============================================================================
# Tests: nested PresetCfg inside PresetCfg
# =============================================================================


def test_collect_nested_presetcfg():
    """PresetCfg inside another PresetCfg's alternatives is discovered."""
    presets = collect_presets(NestedPresetEnvCfg())
    assert "scene" in presets
    assert set(presets["scene"].keys()) == {"default", "with_camera"}
    # camera preset discovered inside with_camera alternative
    assert "scene.camera" in presets
    assert set(presets["scene.camera"].keys()) == {"small", "large", "default"}
    assert isinstance(presets["scene.camera"]["small"], CameraSmallCfg)
    assert isinstance(presets["scene.camera"]["large"], CameraLargeCfg)


def test_nested_presetcfg_pruned_when_parent_has_none():
    """When scene auto-defaults to default (camera=None), nested camera preset is pruned."""
    env_cfg = NestedPresetEnvCfg()
    agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    # No CLI args → scene resolves to default (camera=None), camera preset must NOT apply
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], [], [], presets)
    assert isinstance(env_cfg.scene, BaseSceneCfg)
    assert env_cfg.scene.camera is None


def test_nested_presetcfg_auto_default_with_camera():
    """When with_camera scene is selected, camera auto-defaults to small (the default)."""
    env_cfg = NestedPresetEnvCfg()
    agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    # Only select with_camera scene, camera should auto-default to small
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, ["with_camera"], [], [], presets)
    assert isinstance(env_cfg.scene, BaseSceneCfg)
    assert isinstance(env_cfg.scene.camera, CameraSmallCfg)
    assert env_cfg.scene.camera.width == 64


def test_nested_presetcfg_global_broadcast():
    """Global preset resolves both outer and nested PresetCfg."""
    env_cfg = NestedPresetEnvCfg()
    agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    # "with_camera" selects the scene, "large" selects the camera
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, ["with_camera", "large"], [], [], presets)
    assert isinstance(env_cfg.scene, BaseSceneCfg)
    assert isinstance(env_cfg.scene.camera, CameraLargeCfg)
    assert env_cfg.scene.camera.width == 256


def test_nested_presetcfg_path_selection():
    """Path selection on nested PresetCfg resolves correctly."""
    env_cfg = NestedPresetEnvCfg()
    agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    sel = [("env", "scene", "with_camera"), ("env", "scene.camera", "large")]
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], sel, [], presets)
    assert isinstance(env_cfg.scene, BaseSceneCfg)
    assert isinstance(env_cfg.scene.camera, CameraLargeCfg)
    assert env_cfg.scene.camera.width == 256


# =============================================================================
# Tests: root-level PresetCfg with nested PresetCfg inside alternatives
# (mirrors CartpoleCameraPresetsEnvCfg structure)
# =============================================================================


@configclass
class RendererACfg:
    backend: str = "rtx"


@configclass
class RendererBCfg:
    backend: str = "warp"


@configclass
class RendererPresetCfg(PresetCfg):
    default: RendererACfg = RendererACfg()
    newton: RendererBCfg = RendererBCfg()


@configclass
class SensorBaseCfg:
    data_types: list[str] = []
    width: int = 100
    height: int = 100
    renderer: RendererPresetCfg = RendererPresetCfg()


@configclass
class SensorPresetCfg(PresetCfg):
    default: SensorBaseCfg = SensorBaseCfg(data_types=["rgb"])
    depth: SensorBaseCfg = SensorBaseCfg(data_types=["depth"])


@configclass
class RootEnvBaseCfg:
    decimation: int = 2
    sensor: SensorPresetCfg = SensorPresetCfg()
    obs_shape: list[int] = [100, 100, 3]


@configclass
class RootPresetEnvCfg(PresetCfg):
    default: RootEnvBaseCfg = RootEnvBaseCfg()
    depth: RootEnvBaseCfg = RootEnvBaseCfg(obs_shape=[100, 100, 1])


def test_root_presetcfg_with_nested_preset_collect():
    """collect_presets discovers nested PresetCfg inside root PresetCfg alternatives."""
    presets = collect_presets(RootPresetEnvCfg())
    assert "" in presets
    assert set(presets[""].keys()) == {"default", "depth"}
    assert "sensor" in presets
    assert set(presets["sensor"].keys()) == {"default", "depth"}
    assert "sensor.renderer" in presets
    assert set(presets["sensor.renderer"].keys()) == {"default", "newton"}


def test_root_presetcfg_resolve_defaults():
    """resolve_preset_defaults resolves nested PresetCfg inside root."""
    resolved = resolve_preset_defaults(RootPresetEnvCfg())
    assert isinstance(resolved, RootEnvBaseCfg)
    assert isinstance(resolved.sensor, SensorBaseCfg)
    assert resolved.sensor.data_types == ["rgb"]
    assert isinstance(resolved.sensor.renderer, RendererACfg)
    assert resolved.sensor.renderer.backend == "rtx"


def test_root_presetcfg_global_depth_resolves_nested():
    """Global preset=depth on root PresetCfg also resolves nested sensor and renderer."""
    env_cfg = RootPresetEnvCfg()
    agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}

    env_cfg = resolve_preset_defaults(env_cfg)
    agent_cfg_resolved = resolve_preset_defaults(agent_cfg)

    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg_resolved.to_dict()}

    env_cfg, agent_cfg = apply_overrides(env_cfg, agent_cfg_resolved, hydra_cfg, ["depth"], [], [], presets)

    assert isinstance(env_cfg, RootEnvBaseCfg)
    assert env_cfg.obs_shape == [100, 100, 1]
    assert isinstance(env_cfg.sensor, SensorBaseCfg), (
        f"sensor should be SensorBaseCfg, got {type(env_cfg.sensor).__name__}"
    )
    assert env_cfg.sensor.data_types == ["depth"]
    assert isinstance(env_cfg.sensor.renderer, RendererACfg), (
        f"renderer should be RendererACfg (default), got {type(env_cfg.sensor.renderer).__name__}"
    )
