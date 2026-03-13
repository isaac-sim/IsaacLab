# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
    preset,
    resolve_preset_defaults,
)

# =============================================================================
# Leaf config classes (reused across all test sections)
# =============================================================================


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
class FastPolicyCfg:
    actor_hidden_dims: list = [32, 16]


# =============================================================================
# Composite configs using PresetCfg
# =============================================================================


@configclass
class SampleEnvCfg:
    decimation: int = 4
    sim_dt: float = 0.005


@configclass
class SampleAgentCfg:
    max_iterations: int = 1000
    learning_rate: float = 3e-4


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
def class_presets():
    """Fresh configs using PresetCfg pattern."""
    env_cfg = PresetCfgEnvCfg()
    agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    return env_cfg, agent_cfg, presets


# =============================================================================
# Tests: collect_presets
# =============================================================================


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


# =============================================================================
# Tests: parse_overrides
# =============================================================================


def test_parse_overrides_mixed():
    """All override types categorized correctly."""
    env_cfg = PresetCfgEnvCfg()
    presets = {"env": collect_presets(env_cfg), "agent": {}}
    args = [
        "presets=fast",
        "env.decimation=10",
        "env.backend=newton",
        "env.backend.dt=0.001",
    ]
    global_p, sel, scalar, glob = parse_overrides(args, presets)
    assert global_p == ["fast"]
    assert ("env", "backend", "newton") in sel
    assert ("env.backend.dt", "0.001") in scalar
    assert "env.decimation=10" in glob


def test_parse_overrides_root_preset():
    """Root-level PresetCfg parsed as agent=<name>."""
    presets = {"env": {}, "agent": collect_presets(RootAgentCfg())}
    _, sel, _, _ = parse_overrides(["agent=fast"], presets)
    assert sel == [("agent", "", "fast")]


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


@configclass
class OptionalFeatureCfg:
    buffer_size: int = 200
    export_path: str = "."


@configclass
class OptionalFeaturePresetCfg(PresetCfg):
    default = None
    enabled: OptionalFeatureCfg = OptionalFeatureCfg()


@configclass
class EnvWithOptionalFeatureCfg:
    decimation: int = 4
    optional_feature: OptionalFeaturePresetCfg = OptionalFeaturePresetCfg()


def test_presetcfg_none_default_auto_applies():
    """PresetCfg with default=None auto-applies None without crashing."""
    env_cfg = EnvWithOptionalFeatureCfg()
    agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], [], [], presets)
    assert env_cfg.optional_feature is None


def test_presetcfg_none_default_cli_selects_enabled():
    """PresetCfg with default=None can be overridden to a real config via CLI."""
    env_cfg = EnvWithOptionalFeatureCfg()
    agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    sel = [("env", "optional_feature", "enabled")]
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], sel, [], presets)
    assert isinstance(env_cfg.optional_feature, OptionalFeatureCfg)
    assert env_cfg.optional_feature.buffer_size == 200


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


# =============================================================================
# Tests: scalar PresetCfg (e.g., armature=PresetCfg(default=0.0, newton=0.01))
# =============================================================================


@configclass
class ScalarPresetCfg(PresetCfg):
    default: float = 0.0
    newton: float = 0.01


@configclass
class ActuatorWithScalarPresetCfg:
    joint_names: list = [".*"]
    stiffness: float = 40.0
    damping: float = 5.0
    armature: ScalarPresetCfg = ScalarPresetCfg()


@configclass
class ScalarPresetEnvCfg:
    decimation: int = 4
    actuator: ActuatorWithScalarPresetCfg = ActuatorWithScalarPresetCfg()


def test_scalar_presetcfg_collect():
    """Scalar PresetCfg fields collected with correct values."""
    presets = collect_presets(ScalarPresetEnvCfg())
    assert "actuator.armature" in presets
    assert presets["actuator.armature"]["default"] == 0.0
    assert presets["actuator.armature"]["newton"] == 0.01


def test_scalar_presetcfg_resolve_default():
    """resolve_preset_defaults replaces scalar PresetCfg with its default value."""
    cfg = ScalarPresetEnvCfg()
    resolved = resolve_preset_defaults(cfg)
    assert resolved.actuator.armature == 0.0
    assert not isinstance(resolved.actuator.armature, PresetCfg)


def test_scalar_presetcfg_auto_default():
    """Scalar PresetCfg auto-applies default=0.0 when no CLI override."""
    env_cfg = ScalarPresetEnvCfg()
    agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    env_cfg = resolve_preset_defaults(env_cfg)
    agent_cfg = resolve_preset_defaults(agent_cfg)
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], [], [], presets)
    assert env_cfg.actuator.armature == 0.0


def test_scalar_presetcfg_global_newton():
    """Global preset=newton replaces scalar PresetCfg with newton value."""
    env_cfg = ScalarPresetEnvCfg()
    agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    env_cfg = resolve_preset_defaults(env_cfg)
    agent_cfg = resolve_preset_defaults(agent_cfg)
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, ["newton"], [], [], presets)
    assert env_cfg.actuator.armature == 0.01


def test_scalar_presetcfg_path_selection():
    """Path selection replaces scalar PresetCfg with chosen value."""
    env_cfg = ScalarPresetEnvCfg()
    agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    env_cfg = resolve_preset_defaults(env_cfg)
    agent_cfg = resolve_preset_defaults(agent_cfg)
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    sel = [("env", "actuator.armature", "newton")]
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], sel, [], presets)
    assert env_cfg.actuator.armature == 0.01
    assert env_cfg.actuator.stiffness == 40.0  # other fields untouched


# =============================================================================
# Tests: PresetCfg inside dict values (e.g., actuators["legs"].armature)
# =============================================================================


@configclass
class ActuatorCfgWithPreset:
    joint_names: list = [".*"]
    stiffness: float = 40.0
    damping: float = 5.0
    armature: ScalarPresetCfg = ScalarPresetCfg()


@configclass
class RobotCfg:
    prim_path: str = "/World/Robot"
    actuators: dict = None

    def __post_init__(self):
        if self.actuators is None:
            self.actuators = {"legs": ActuatorCfgWithPreset()}


@configclass
class DictPresetEnvCfg:
    decimation: int = 4
    robot: RobotCfg = RobotCfg()


def test_collect_presets_traverses_dict_values():
    """collect_presets finds PresetCfg inside dict-held configclass values."""
    cfg = DictPresetEnvCfg()
    presets = collect_presets(cfg)
    assert "robot.actuators.legs.armature" in presets
    assert presets["robot.actuators.legs.armature"]["default"] == 0.0
    assert presets["robot.actuators.legs.armature"]["newton"] == 0.01


def test_resolve_preset_defaults_traverses_dict_values():
    """resolve_preset_defaults resolves PresetCfg inside dict-held configclass values."""
    cfg = DictPresetEnvCfg()
    resolved = resolve_preset_defaults(cfg)
    assert resolved.robot.actuators["legs"].armature == 0.0
    assert not isinstance(resolved.robot.actuators["legs"].armature, PresetCfg)


def test_dict_preset_auto_default():
    """Dict-held PresetCfg auto-applies default when no CLI override."""
    env_cfg = DictPresetEnvCfg()
    agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    env_cfg = resolve_preset_defaults(env_cfg)
    agent_cfg = resolve_preset_defaults(agent_cfg)
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], [], [], presets)
    assert env_cfg.robot.actuators["legs"].armature == 0.0


def test_dict_preset_global_newton():
    """Global preset=newton replaces dict-held scalar PresetCfg."""
    env_cfg = DictPresetEnvCfg()
    agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    env_cfg = resolve_preset_defaults(env_cfg)
    agent_cfg = resolve_preset_defaults(agent_cfg)
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, ["newton"], [], [], presets)
    assert env_cfg.robot.actuators["legs"].armature == 0.01


def test_dict_preset_path_selection():
    """Path selection replaces dict-held scalar PresetCfg."""
    env_cfg = DictPresetEnvCfg()
    agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    env_cfg = resolve_preset_defaults(env_cfg)
    agent_cfg = resolve_preset_defaults(agent_cfg)
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    sel = [("env", "robot.actuators.legs.armature", "newton")]
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], sel, [], presets)
    assert env_cfg.robot.actuators["legs"].armature == 0.01
    assert env_cfg.robot.actuators["legs"].stiffness == 40.0


def test_dict_preset_with_factory():
    """preset() factory works inside dict-held configclass values."""

    @configclass
    class ActuatorCfgFactory:
        joint_names: list = [".*"]
        armature: object = None

        def __post_init__(self):
            if self.armature is None:
                self.armature = preset(default=0.0, newton=0.01, physx=0.0)

    @configclass
    class RobotCfgFactory:
        actuators: dict = None

        def __post_init__(self):
            if self.actuators is None:
                self.actuators = {"legs": ActuatorCfgFactory()}

    @configclass
    class EnvCfgFactory:
        robot: RobotCfgFactory = RobotCfgFactory()

    cfg = EnvCfgFactory()
    presets = collect_presets(cfg)
    assert "robot.actuators.legs.armature" in presets
    assert presets["robot.actuators.legs.armature"]["default"] == 0.0
    assert presets["robot.actuators.legs.armature"]["newton"] == 0.01
    assert presets["robot.actuators.legs.armature"]["physx"] == 0.0
