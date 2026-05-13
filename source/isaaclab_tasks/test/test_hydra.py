# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Self-contained tests for Hydra configuration utilities.

These tests verify the REPLACE-only preset system without depending on
external environment configurations.
"""

import warnings

import pytest

from isaaclab.utils import configclass

from isaaclab_tasks.utils import hydra as hydra_mod
from isaaclab_tasks.utils.hydra import (
    PresetCfg,
    _format_unknown_presets_error,
    apply_overrides,
    collect_presets,
    parse_overrides,
    preset,
    resolve_presets,
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
    newton_mjwarp: NewtonCfg = NewtonCfg()


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
    """Root-level PresetCfg -- the agent config itself is a PresetCfg."""

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


# -- Scalar PresetCfg and actuator configs (shared by scalar + dict sections) --


@configclass
class ScalarPresetCfg(PresetCfg):
    default: float = 0.0
    newton_mjwarp: float = 0.01


@configclass
class ActuatorWithPresetCfg:
    joint_names: list = [".*"]
    stiffness: float = 40.0
    damping: float = 5.0
    armature: ScalarPresetCfg = ScalarPresetCfg()


# -- Deep-nested dict configs (event term params pattern) --


@configclass
class OffsetCfg(PresetCfg):
    """Mimics task-specific offset presets (e.g., AssembledOffsetCfg)."""

    task_a: tuple = (0.0, 0.0, 0.01)
    task_b: tuple = (0.02, 0.0, 0.005)
    default: tuple = task_a


@configclass
class FractionCfg(PresetCfg):
    task_a: tuple = (0.05, 0.5)
    task_b: tuple = (0.3, 1.0)
    default: tuple = task_a


@configclass
class JointNamesCfg(PresetCfg):
    default: list[str] | None = None
    robot_a: list[str] = None
    robot_b: list[str] = None


@configclass
class EntityCfg:
    """Mimics SceneEntityCfg with a preset-valued field."""

    name: str = "robot"
    joint_names: list[str] | None = None


@configclass
class InnerTermCfg:
    """Mimics an EventTermCfg with params containing presets."""

    func: str = "reset_fn"
    params: dict = None

    def __post_init__(self):
        if self.params is None:
            self.params = {
                "offset": OffsetCfg(),
                "fraction": FractionCfg(),
                "robot_cfg": EntityCfg(name="robot", joint_names=JointNamesCfg()),
            }


@configclass
class OuterTermCfg:
    """Mimics a chained reset term with nested terms dict."""

    func: str = "chain_fn"
    params: dict = None

    def __post_init__(self):
        if self.params is None:
            self.params = {
                "terms": {
                    "step_one": InnerTermCfg(),
                }
            }


@configclass
class DeepDictEnvCfg:
    decimation: int = 4
    events: OuterTermCfg = OuterTermCfg()


@configclass
class DictPresetTermCfg:
    """Outer term where the terms dict is itself a preset (resolves to a dict)."""

    func: str = "term_choice"
    params: dict = None

    def __post_init__(self):
        if self.params is None:
            self.params = {
                "terms": preset(
                    default={
                        "strategy_a": InnerTermCfg(),
                        "strategy_b": InnerTermCfg(),
                    },
                    alt={
                        "strategy_a": InnerTermCfg(),
                    },
                ),
            }


@configclass
class PresetResolvesToDictEnvCfg:
    decimation: int = 4
    events: DictPresetTermCfg = DictPresetTermCfg()


# =============================================================================
# Helpers
# =============================================================================


def _apply(env_cfg, agent_cfg=None, global_presets=None, preset_sel=None, preset_scalar=None):
    """Collect presets, resolve defaults, build hydra dict, and apply overrides."""
    if agent_cfg is None:
        agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    env_cfg = resolve_presets(env_cfg)
    agent_cfg = resolve_presets(agent_cfg)
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    return apply_overrides(
        env_cfg,
        agent_cfg,
        hydra_cfg,
        global_presets or [],
        preset_sel or [],
        preset_scalar or [],
        presets,
    )


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
    assert set(presets["backend"].keys()) == {"default", "newton_mjwarp"}
    assert isinstance(presets["backend"]["default"], PhysxCfg)
    assert isinstance(presets["backend"]["newton_mjwarp"], NewtonCfg)


def test_legacy_newton_attribute_alias_warns():
    """Python access to the legacy ``newton`` preset aliases to ``newton_mjwarp`` during deprecation."""
    cfg = SimBackendCfg()
    with pytest.warns(FutureWarning, match="Preset 'newton' is deprecated"):
        assert cfg.newton is cfg.newton_mjwarp


def test_legacy_kamino_attribute_alias_warns():
    """Python access to the legacy ``kamino`` preset aliases to ``newton_kamino`` during deprecation."""

    @configclass
    class _SolverPresetsCfg(PresetCfg):
        default: PhysxCfg = PhysxCfg()
        newton_kamino: NewtonCfg = NewtonCfg()

    cfg = _SolverPresetsCfg()
    with pytest.warns(FutureWarning, match="Preset 'kamino' is deprecated"):
        assert cfg.kamino is cfg.newton_kamino


def test_legacy_alias_suppressed_when_legacy_name_is_real_field():
    """An env that legitimately defines ``newton`` should not warn or be remapped."""

    @configclass
    class _ShadowingCfg(PresetCfg):
        default: PhysxCfg = PhysxCfg()
        newton: PhysxCfg = PhysxCfg()
        newton_mjwarp: NewtonCfg = NewtonCfg()

    cfg = _ShadowingCfg()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        assert cfg.newton is not cfg.newton_mjwarp
        assert isinstance(cfg.newton, PhysxCfg)


def test_presetcfg_attribute_error_for_unknown_attribute():
    """Plain missing attributes should raise ``AttributeError`` (not warn or alias)."""
    cfg = SimBackendCfg()
    assert not hasattr(cfg, "completely_unknown")
    with pytest.raises(AttributeError, match="completely_unknown"):
        _ = cfg.completely_unknown


def test_format_unknown_presets_error_calls_out_legacy_aliases():
    """The unknown-preset error should explicitly mention the rename for legacy aliases."""
    msg = _format_unknown_presets_error({"newton", "typo"}, {"fast": ["env"]})
    assert "newton' was renamed to 'newton_mjwarp'" in msg
    assert "typo" in msg


def test_user_stacklevel_warning_origin_is_outside_hydra_module():
    """``_normalize_preset_name`` warnings should not be attributed to hydra.py itself."""
    presets_arg = {"env": {"backend": {"default": None, "newton_mjwarp": None}}, "agent": {}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        parse_overrides(["presets=newton"], presets_arg)
    deprecations = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert deprecations, "expected a FutureWarning from the legacy alias"
    assert deprecations[0].filename != hydra_mod.__file__, (
        f"warning was attributed to hydra.py ({deprecations[0].filename}); _user_stacklevel should "
        f"point outside the module"
    )


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
        "env.backend=newton_mjwarp",
        "env.backend.dt=0.001",
    ]
    global_p, sel, scalar, glob = parse_overrides(args, presets)
    assert global_p == ["fast"]
    assert ("env", "backend", "newton_mjwarp") in sel
    assert ("env.backend.dt", "0.001") in scalar
    assert "env.decimation=10" in glob


def test_parse_overrides_root_preset():
    """Root-level PresetCfg parsed as agent=<name>."""
    presets = {"env": {}, "agent": collect_presets(RootAgentCfg())}
    _, sel, _, _ = parse_overrides(["agent=fast"], presets)
    assert sel == [("agent", "", "fast")]


# =============================================================================
# Tests: apply_overrides -- PresetCfg (nested + broadcast + root)
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
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], [("env", "backend", "newton_mjwarp")], [], presets)
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
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], [("env", "backend", "newton_mjwarp")], [], presets)
    assert isinstance(env_cfg.backend, NewtonCfg)
    assert isinstance(env_cfg.observations, NoiselessObservationsCfg)
    assert isinstance(agent_cfg.policy, SmallPolicyCfg)


def test_root_presetcfg_auto_default():
    """Root-level PresetCfg auto-applies 'default'."""
    env_cfg, agent_cfg = _apply(SampleEnvCfg(), RootAgentCfg())
    assert isinstance(agent_cfg, SampleAgentCfg)
    assert agent_cfg.max_iterations == 1000


def test_root_presetcfg_cli_selection():
    """Root-level PresetCfg resolved via path selection."""
    env_cfg, agent_cfg = _apply(SampleEnvCfg(), RootAgentCfg(), preset_sel=[("agent", "", "fast")])
    assert isinstance(agent_cfg, SampleAgentCfg)
    assert agent_cfg.max_iterations == 100
    assert agent_cfg.learning_rate == 1e-3


def test_root_presetcfg_global_preset():
    """Root-level PresetCfg resolved via global preset."""
    env_cfg, agent_cfg = _apply(SampleEnvCfg(), RootAgentCfg(), global_presets=["fast"])
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
    assert "scene.camera" in presets
    assert set(presets["scene.camera"].keys()) == {"small", "large", "default"}
    assert isinstance(presets["scene.camera"]["small"], CameraSmallCfg)
    assert isinstance(presets["scene.camera"]["large"], CameraLargeCfg)


def test_nested_presetcfg_pruned_when_parent_has_none():
    """When scene auto-defaults to default (camera=None), nested camera preset is pruned."""
    env_cfg, _ = _apply(NestedPresetEnvCfg())
    assert isinstance(env_cfg.scene, BaseSceneCfg)
    assert env_cfg.scene.camera is None


def test_nested_presetcfg_auto_default_with_camera():
    """When with_camera scene is selected, camera auto-defaults to small (the default)."""
    env_cfg, _ = _apply(NestedPresetEnvCfg(), global_presets=["with_camera"])
    assert isinstance(env_cfg.scene, BaseSceneCfg)
    assert isinstance(env_cfg.scene.camera, CameraSmallCfg)
    assert env_cfg.scene.camera.width == 64


def test_nested_presetcfg_global_broadcast():
    """Global preset resolves both outer and nested PresetCfg."""
    env_cfg, _ = _apply(NestedPresetEnvCfg(), global_presets=["with_camera", "large"])
    assert isinstance(env_cfg.scene, BaseSceneCfg)
    assert isinstance(env_cfg.scene.camera, CameraLargeCfg)
    assert env_cfg.scene.camera.width == 256


def test_nested_presetcfg_path_selection():
    """Path selection on nested PresetCfg resolves correctly."""
    sel = [("env", "scene", "with_camera"), ("env", "scene.camera", "large")]
    env_cfg, _ = _apply(NestedPresetEnvCfg(), preset_sel=sel)
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
    newton_renderer: RendererBCfg = RendererBCfg()


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
    assert set(presets["sensor.renderer"].keys()) == {"default", "newton_renderer"}


def test_root_presetcfg_resolve_defaults():
    """resolve_presets resolves nested PresetCfg inside root."""
    resolved = resolve_presets(RootPresetEnvCfg())
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
    env_cfg, _ = _apply(EnvWithOptionalFeatureCfg())
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
    env_cfg, _ = _apply(RootPresetEnvCfg(), global_presets=["depth"])
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
# Tests: scalar PresetCfg (e.g., armature=PresetCfg(default=0.0, newton_mjwarp=0.01))
# =============================================================================


@configclass
class ScalarPresetEnvCfg:
    decimation: int = 4
    actuator: ActuatorWithPresetCfg = ActuatorWithPresetCfg()


def test_scalar_presetcfg_collect():
    """Scalar PresetCfg fields collected with correct values."""
    presets = collect_presets(ScalarPresetEnvCfg())
    assert "actuator.armature" in presets
    assert presets["actuator.armature"]["default"] == 0.0
    assert presets["actuator.armature"]["newton_mjwarp"] == 0.01


def test_scalar_presetcfg_resolve_default():
    """resolve_presets replaces scalar PresetCfg with its default value."""
    cfg = ScalarPresetEnvCfg()
    resolved = resolve_presets(cfg)
    assert resolved.actuator.armature == 0.0
    assert not isinstance(resolved.actuator.armature, PresetCfg)


def test_scalar_presetcfg_auto_default():
    """Scalar PresetCfg auto-applies default=0.0 when no CLI override."""
    env_cfg, _ = _apply(ScalarPresetEnvCfg())
    assert env_cfg.actuator.armature == 0.0


def test_scalar_presetcfg_global_newton_mjwarp():
    """Global preset=newton_mjwarp replaces scalar PresetCfg with MJWarp value."""
    env_cfg, _ = _apply(ScalarPresetEnvCfg(), global_presets=["newton_mjwarp"])
    assert env_cfg.actuator.armature == 0.01


def test_scalar_presetcfg_path_selection():
    """Path selection replaces scalar PresetCfg with chosen value."""
    env_cfg, _ = _apply(ScalarPresetEnvCfg(), preset_sel=[("env", "actuator.armature", "newton_mjwarp")])
    assert env_cfg.actuator.armature == 0.01
    assert env_cfg.actuator.stiffness == 40.0


# =============================================================================
# Tests: PresetCfg inside dict values (e.g., actuators["legs"].armature)
# =============================================================================


@configclass
class RobotCfg:
    prim_path: str = "/World/Robot"
    actuators: dict = None

    def __post_init__(self):
        if self.actuators is None:
            self.actuators = {"legs": ActuatorWithPresetCfg()}


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
    assert presets["robot.actuators.legs.armature"]["newton_mjwarp"] == 0.01


def test_resolve_presets_traverses_dict_values():
    """resolve_presets resolves PresetCfg inside dict-held configclass values."""
    cfg = DictPresetEnvCfg()
    resolved = resolve_presets(cfg)
    assert resolved.robot.actuators["legs"].armature == 0.0
    assert not isinstance(resolved.robot.actuators["legs"].armature, PresetCfg)


def test_dict_preset_auto_default():
    """Dict-held PresetCfg auto-applies default when no CLI override."""
    env_cfg, _ = _apply(DictPresetEnvCfg())
    assert env_cfg.robot.actuators["legs"].armature == 0.0


def test_dict_preset_global_newton_mjwarp():
    """Global preset=newton_mjwarp replaces dict-held scalar PresetCfg."""
    env_cfg, _ = _apply(DictPresetEnvCfg(), global_presets=["newton_mjwarp"])
    assert env_cfg.robot.actuators["legs"].armature == 0.01


def test_dict_preset_path_selection():
    """Path selection replaces dict-held scalar PresetCfg."""
    env_cfg, _ = _apply(DictPresetEnvCfg(), preset_sel=[("env", "robot.actuators.legs.armature", "newton_mjwarp")])
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
                self.armature = preset(default=0.0, newton_mjwarp=0.01, physx=0.0)

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
    assert presets["robot.actuators.legs.armature"]["newton_mjwarp"] == 0.01
    assert presets["robot.actuators.legs.armature"]["physx"] == 0.0


# =============================================================================
# Tests: rough terrain config regressions
# =============================================================================


def test_go1_rough_newton_mjwarp_armature_preset():
    """Go1 rough terrain uses higher MJWarp armature without changing PhysX."""
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.rough_env_cfg import UnitreeGo1RoughEnvCfg

    env_cfg, _ = _apply(UnitreeGo1RoughEnvCfg(), global_presets=["newton_mjwarp"])
    assert env_cfg.scene.robot.actuators["base_legs"].armature == 0.02

    env_cfg, _ = _apply(UnitreeGo1RoughEnvCfg())
    assert env_cfg.scene.robot.actuators["base_legs"].armature == 0.0


def test_go1_rough_legacy_newton_alias_resolves_to_newton_mjwarp():
    """Real-config alias path: ``presets=newton`` against an actual env cfg resolves to newton_mjwarp."""
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.rough_env_cfg import UnitreeGo1RoughEnvCfg

    with pytest.warns(FutureWarning, match="Preset 'newton' is deprecated"):
        env_cfg, _ = _apply(UnitreeGo1RoughEnvCfg(), global_presets=["newton"])
    assert env_cfg.scene.robot.actuators["base_legs"].armature == 0.02


# =============================================================================
# Tests: PresetCfg inside deeply nested dicts (e.g., event term params)
# =============================================================================


def test_collect_presets_deep_nested_dicts():
    """collect_presets discovers PresetCfg inside dict->dict->configclass->dict chains."""
    cfg = DeepDictEnvCfg()
    presets = collect_presets(cfg)
    offset_path = "events.params.terms.step_one.params.offset"
    fraction_path = "events.params.terms.step_one.params.fraction"
    assert offset_path in presets, f"Expected '{offset_path}' in {list(presets.keys())}"
    assert fraction_path in presets, f"Expected '{fraction_path}' in {list(presets.keys())}"
    assert presets[offset_path]["task_a"] == (0.0, 0.0, 0.01)
    assert presets[offset_path]["task_b"] == (0.02, 0.0, 0.005)
    assert presets[fraction_path]["task_a"] == (0.05, 0.5)
    assert presets[fraction_path]["task_b"] == (0.3, 1.0)


def test_resolve_presets_deep_nested_dicts():
    """resolve_presets resolves presets inside deeply nested dicts."""
    cfg = DeepDictEnvCfg()
    resolved = resolve_presets(cfg)
    inner = resolved.events.params["terms"]["step_one"]
    assert inner.params["offset"] == (0.0, 0.0, 0.01)
    assert inner.params["fraction"] == (0.05, 0.5)
    assert not isinstance(inner.params["offset"], PresetCfg)
    assert not isinstance(inner.params["fraction"], PresetCfg)
    assert inner.params["robot_cfg"].joint_names is None
    assert not isinstance(inner.params["robot_cfg"].joint_names, PresetCfg)


def test_deep_nested_dict_auto_default():
    """Deeply nested dict presets auto-apply default when no CLI override."""
    env_cfg, _ = _apply(DeepDictEnvCfg())
    inner = env_cfg.events.params["terms"]["step_one"]
    assert inner.params["offset"] == (0.0, 0.0, 0.01)
    assert inner.params["fraction"] == (0.05, 0.5)


def test_deep_nested_dict_global_preset():
    """Global preset=task_b replaces deeply nested dict presets."""
    env_cfg, _ = _apply(DeepDictEnvCfg(), global_presets=["task_b"])
    inner = env_cfg.events.params["terms"]["step_one"]
    assert inner.params["offset"] == (0.02, 0.0, 0.005), f"offset should be task_b value, got {inner.params['offset']}"
    assert inner.params["fraction"] == (0.3, 1.0), f"fraction should be task_b value, got {inner.params['fraction']}"


def test_deep_nested_dict_path_selection():
    """Path selection replaces a specific deeply nested dict preset."""
    sel = [("env", "events.params.terms.step_one.params.offset", "task_b")]
    env_cfg, _ = _apply(DeepDictEnvCfg(), preset_sel=sel)
    inner = env_cfg.events.params["terms"]["step_one"]
    assert inner.params["offset"] == (0.02, 0.0, 0.005)
    assert inner.params["fraction"] == (0.05, 0.5)


def test_deep_nested_dict_mixed_global_and_path():
    """Global preset applies to nested dicts, path selection overrides one."""
    sel = [("env", "events.params.terms.step_one.params.fraction", "task_a")]
    env_cfg, _ = _apply(DeepDictEnvCfg(), global_presets=["task_b"], preset_sel=sel)
    inner = env_cfg.events.params["terms"]["step_one"]
    assert inner.params["offset"] == (0.02, 0.0, 0.005)
    assert inner.params["fraction"] == (0.05, 0.5)


# =============================================================================
# Tests: preset resolving to dict containing further presets
# =============================================================================


def test_collect_presets_discovers_presets_inside_dict_valued_alternatives():
    """collect_presets must recurse into dict-valued preset alternatives to
    discover further PresetCfg nodes nested inside them.
    """
    cfg = PresetResolvesToDictEnvCfg()
    presets = collect_presets(cfg)
    offset_paths = [p for p in presets if "offset" in p]
    fraction_paths = [p for p in presets if "fraction" in p]
    assert len(offset_paths) > 0, (
        f"OffsetCfg inside dict-valued preset alternative not discovered. Found: {list(presets.keys())}"
    )
    assert len(fraction_paths) > 0, (
        f"FractionCfg inside dict-valued preset alternative not discovered. Found: {list(presets.keys())}"
    )


def test_resolve_preset_resolving_to_dict_walks_contents():
    """When a preset resolves to a dict, presets inside that dict are also resolved.

    Also verifies that PresetCfg(default=None) nested inside the resolved dict
    correctly resolves to None (not skipped).
    """
    cfg = PresetResolvesToDictEnvCfg()
    resolved = resolve_presets(cfg)

    terms = resolved.events.params["terms"]
    assert isinstance(terms, dict), f"Expected dict, got {type(terms)}"
    assert not isinstance(terms, PresetCfg), "Top-level preset was not resolved"

    for name, term in terms.items():
        entity = term.params["robot_cfg"]
        assert not isinstance(entity.joint_names, PresetCfg), (
            f"PresetCfg leaked into {name}.params.robot_cfg.joint_names"
        )
        assert entity.joint_names is None
        assert not isinstance(term.params["offset"], PresetCfg)
        assert not isinstance(term.params["fraction"], PresetCfg)


def test_resolve_preset_uses_class_level_override():
    """When a robot-specific module overrides PresetCfg.default at class level
    after instances are created, resolve_presets picks up the override."""

    @configclass
    class BodyNameCfg(PresetCfg):
        default: str = "generic_body"

    @configclass
    class TermWithBody:
        func: str = "some_fn"
        params: dict = None

        def __post_init__(self):
            if self.params is None:
                self.params = {"cfg": EntityCfg(name="robot", joint_names=BodyNameCfg())}

    @configclass
    class EnvWithBody:
        events: TermWithBody = TermWithBody()

    BodyNameCfg.default = "robot_specific_body"

    cfg = EnvWithBody()
    resolved = resolve_presets(cfg)
    assert resolved.events.params["cfg"].joint_names == "robot_specific_body"
    assert not isinstance(resolved.events.params["cfg"].joint_names, PresetCfg)


def test_resolve_presets_with_selected_name_in_deeply_nested_dict():
    """resolve_presets(cfg, {"task_b"}) must select task_b alternatives
    for PresetCfg instances nested inside dict-valued preset alternatives.
    """
    cfg = PresetResolvesToDictEnvCfg()
    resolved = resolve_presets(cfg, {"task_b"})

    terms = resolved.events.params["terms"]
    assert isinstance(terms, dict)
    for name, term in terms.items():
        assert term.params["offset"] == (0.02, 0.0, 0.005), (
            f"{name}: offset should be task_b, got {term.params['offset']}"
        )
        assert term.params["fraction"] == (0.3, 1.0), (
            f"{name}: fraction should be task_b, got {term.params['fraction']}"
        )


# =============================================================================
# Tests: preset() factory function
# =============================================================================


def test_preset_factory_creates_presetcfg():
    """preset() returns a PresetCfg subclass instance with correct fields."""
    p = preset(default=0.0, high=1.0, low=-1.0)
    assert isinstance(p, PresetCfg)
    assert p.default == 0.0
    assert p.high == 1.0
    assert p.low == -1.0


def test_preset_factory_collectable():
    """preset()-created instances are discovered by collect_presets."""

    @configclass
    class FactoryEnvCfg:
        damping: object = None

        def __post_init__(self):
            if self.damping is None:
                self.damping = preset(default=5.0, high=20.0)

    cfg = FactoryEnvCfg()
    presets = collect_presets(cfg)
    assert "damping" in presets
    assert presets["damping"]["default"] == 5.0
    assert presets["damping"]["high"] == 20.0


def test_preset_factory_requires_default():
    """preset() raises ValueError when 'default' is not provided."""
    with pytest.raises(ValueError, match="default"):
        preset(high=1.0, low=-1.0)


def test_preset_factory_string_values():
    """preset() works with string values."""
    p = preset(default="cpu", gpu="cuda:0")
    assert isinstance(p, PresetCfg)
    assert p.default == "cpu"
    assert p.gpu == "cuda:0"


# =============================================================================
# Tests: _collect_fields class-vs-instance priority
# =============================================================================


def test_collect_fields_prefers_class_attr_over_instance():
    """Class-level attr mutations take priority over instance attrs in collection.

    This mirrors the pattern where robot-specific modules (e.g., joint_pos_env_cfg.py)
    mutate PresetCfg class attributes after instances are already created.
    """

    @configclass
    class MutablePresetCfg(PresetCfg):
        default: str = "original_default"
        alt: str = "alternative"

    instance = MutablePresetCfg()
    assert instance.default == "original_default"

    MutablePresetCfg.default = "robot_specific_default"

    presets = collect_presets(instance)
    assert "" in presets
    assert presets[""]["default"] == "robot_specific_default"

    MutablePresetCfg.default = "original_default"


def test_collect_fields_includes_dynamic_class_attrs():
    """Fields added to PresetCfg class at runtime are discovered."""

    @configclass
    class ExtensiblePresetCfg(PresetCfg):
        default: str = "base"
        alt_a: str = "a"

    ExtensiblePresetCfg.alt_b = "b"

    instance = ExtensiblePresetCfg()
    presets = collect_presets(instance)
    assert "" in presets
    assert "alt_b" in presets[""]
    assert presets[""]["alt_b"] == "b"

    delattr(ExtensiblePresetCfg, "alt_b")


# =============================================================================
# Tests: apply_overrides error handling
# =============================================================================


def test_apply_overrides_unknown_preset_group_raises():
    """apply_overrides raises ValueError for unknown preset group paths."""
    env_cfg = PresetCfgEnvCfg()
    agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    with pytest.raises(ValueError, match="Unknown preset group"):
        apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], [("env", "nonexistent", "val")], [], presets)


def test_apply_overrides_unknown_preset_name_raises():
    """apply_overrides raises ValueError for unknown preset name."""
    env_cfg = PresetCfgEnvCfg()
    agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    with pytest.raises(ValueError, match="Unknown preset 'nonexistent'"):
        apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], [("env", "backend", "nonexistent")], [], presets)


def test_apply_overrides_conflicting_globals_raises():
    """Two global presets matching the same path cause ValueError."""

    @configclass
    class TwoAltsPresetCfg(PresetCfg):
        default: str = "d"
        opt_a: str = "a"
        opt_b: str = "b"

    @configclass
    class ConflictEnvCfg:
        mode: TwoAltsPresetCfg = TwoAltsPresetCfg()

    env_cfg = ConflictEnvCfg()
    agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    with pytest.raises(ValueError, match="Conflicting global presets"):
        apply_overrides(env_cfg, agent_cfg, hydra_cfg, ["opt_a", "opt_b"], [], [], presets)


def test_apply_overrides_aliased_globals_no_conflict():
    """Two global presets resolving to equal values do not raise.

    Mirrors the dexsuite ObjectCfg pattern where ``newton_mjwarp = cube`` creates
    separate but equal dataclass instances after @configclass processing.
    """

    @configclass
    class SharedCfg:
        value: int = 42

    cube_val = SharedCfg()
    mjwarp_val = SharedCfg()

    @configclass
    class AliasedPresetCfg(PresetCfg):
        default: str = "d"
        cube: SharedCfg = cube_val
        newton_mjwarp: SharedCfg = mjwarp_val

    @configclass
    class AliasedEnvCfg:
        mode: AliasedPresetCfg = AliasedPresetCfg()

    env_cfg = AliasedEnvCfg()
    agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    assert presets["env"]["mode"]["cube"] is not presets["env"]["mode"]["newton_mjwarp"]
    assert presets["env"]["mode"]["cube"] == presets["env"]["mode"]["newton_mjwarp"]
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, ["cube", "newton_mjwarp"], [], [], presets)
    assert env_cfg.mode == SharedCfg()


# =============================================================================
# Tests: parse_overrides edge cases
# =============================================================================


def test_parse_overrides_multiple_global_presets():
    """Multiple comma-separated global presets are split correctly."""
    presets = {"env": {"backend": {"default": None, "newton_mjwarp": None}}, "agent": {}}
    global_p, _, _, _ = parse_overrides(["presets=fast,newton_mjwarp,debug"], presets)
    assert global_p == ["fast", "newton_mjwarp", "debug"]


def test_parse_overrides_maps_legacy_newton_preset_to_newton_mjwarp():
    """Legacy ``newton`` preset selections resolve to ``newton_mjwarp`` when available."""
    presets = {"env": {"backend": {"default": None, "newton_mjwarp": None}}, "agent": {}}
    legacy_name = "newton"

    global_p, sel, _, _ = parse_overrides(["presets=fast," + legacy_name, f"env.backend={legacy_name}"], presets)

    assert global_p == ["fast", "newton_mjwarp"]
    assert sel == [("env", "backend", "newton_mjwarp")]


def test_parse_overrides_maps_legacy_kamino_preset_to_newton_kamino():
    """Legacy ``kamino`` preset selections resolve to ``newton_kamino`` when available."""
    presets = {"env": {"solver": {"default": None, "newton_kamino": None}}, "agent": {}}
    legacy_name = "kamino"

    global_p, sel, _, _ = parse_overrides(["presets=" + legacy_name, f"env.solver={legacy_name}"], presets)

    assert global_p == ["newton_kamino"]
    assert sel == [("env", "solver", "newton_kamino")]


def test_apply_overrides_resolves_legacy_alias_in_global_and_path_selection(class_presets):
    """``apply_overrides`` resolves legacy names supplied directly (bypassing ``parse_overrides``)."""
    env_cfg, agent_cfg, presets = class_presets
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    with pytest.warns(FutureWarning, match="Preset 'newton' is deprecated"):
        apply_overrides(
            env_cfg,
            agent_cfg,
            hydra_cfg,
            global_presets=["newton"],
            preset_sel=[("env", "backend", "newton")],
            preset_scalar=[],
            presets=presets,
        )
    assert isinstance(env_cfg.backend, NewtonCfg)


def test_apply_overrides_legacy_and_current_alias_do_not_conflict(class_presets):
    """``presets=newton,newton_mjwarp`` (legacy + current) resolves to one preset, not a conflict."""
    env_cfg, agent_cfg, presets = class_presets
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    with pytest.warns(FutureWarning, match="Preset 'newton' is deprecated"):
        apply_overrides(env_cfg, agent_cfg, hydra_cfg, ["newton", "newton_mjwarp"], [], [], presets)
    assert isinstance(env_cfg.backend, NewtonCfg)


def test_parse_overrides_no_equals_treated_as_global_scalar():
    """Arguments without '=' are passed through as global scalars."""
    presets = {"env": {}, "agent": {}}
    _, _, _, global_scalar = parse_overrides(["--flag", "positional"], presets)
    assert "--flag" in global_scalar
    assert "positional" in global_scalar


def test_parse_overrides_preset_scalar_detection():
    """Scalar within a preset path is detected as preset_scalar."""
    presets = {"env": {"backend": {"default": None}}, "agent": {}}
    _, _, preset_scalar, _ = parse_overrides(["env.backend.dt=0.001", "env.backend.substeps=4"], presets)
    assert ("env.backend.dt", "0.001") in preset_scalar
    assert ("env.backend.substeps", "4") in preset_scalar


def test_parse_overrides_root_level_env_preset():
    """Root-level PresetCfg (path='') makes env=<name> a valid preset selection."""
    presets = {"env": {"": {"default": None, "fast": None}}, "agent": {}}
    _, sel, _, _ = parse_overrides(["env=fast"], presets)
    assert sel == [("env", "", "fast")]


# =============================================================================
# Tests: _parse_val
# =============================================================================


def test_parse_val_types():
    """_parse_val converts strings to correct Python types."""
    from isaaclab_tasks.utils.hydra import _parse_val

    assert _parse_val("true") is True
    assert _parse_val("True") is True
    assert _parse_val("false") is False
    assert _parse_val("none") is None
    assert _parse_val("null") is None
    assert _parse_val("42") == 42
    assert isinstance(_parse_val("42"), int)
    assert _parse_val("3.14") == 3.14
    assert isinstance(_parse_val("3.14"), float)
    assert _parse_val("hello") == "hello"
    assert _parse_val('"quoted"') == "quoted"
    assert _parse_val("'single'") == "single"


# =============================================================================
# Tests: scalar override within preset path
# =============================================================================


def test_scalar_override_within_preset_path(class_presets):
    """Scalar overrides within preset paths are applied on top of the preset."""
    env_cfg, agent_cfg, presets = class_presets
    env_cfg = resolve_presets(env_cfg)
    agent_cfg = resolve_presets(agent_cfg)
    hydra_cfg = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
    apply_overrides(
        env_cfg,
        agent_cfg,
        hydra_cfg,
        [],
        [("env", "backend", "newton_mjwarp")],
        [("env.backend.dt", "0.001")],
        presets,
    )
    assert isinstance(env_cfg.backend, NewtonCfg)
    assert env_cfg.backend.dt == 0.001
    assert env_cfg.backend.substeps == 4


# =============================================================================
# Tests: resolve_presets idempotency
# =============================================================================


def test_resolve_presets_idempotent():
    """Calling resolve_presets twice yields the same result."""
    cfg = PresetCfgEnvCfg()
    first = resolve_presets(cfg)
    second = resolve_presets(first)
    assert isinstance(second.backend, PhysxCfg)
    assert isinstance(second.observations, NoiselessObservationsCfg)
    assert second.backend.dt == first.backend.dt


def test_unknown_global_preset_name_detected():
    """A selected preset name that doesn't match any PresetCfg field is detected.

    This catches typos like presets=peg_insrt_4mm (missing 'e'). The validation
    in register_task raises ValueError before resolution begins.
    """
    cfg = PresetCfgEnvCfg()
    presets = {"env": collect_presets(cfg), "agent": {}}
    all_known = {name for alts in presets.values() for fields in alts.values() for name in fields if name != "default"}

    assert "newton_mjwarp" in all_known
    assert "typo_preset" not in all_known


def test_resolve_presets_errors_on_no_default():
    """A PresetCfg with no 'default' field and no matching selected name
    must raise ValueError, not silently linger or infinite loop."""

    @configclass
    class NoDefaultPreset(PresetCfg):
        option_a: int = 1

    @configclass
    class EnvCfg:
        mode: NoDefaultPreset = NoDefaultPreset()

    with pytest.raises(ValueError, match="no 'default' field"):
        resolve_presets(EnvCfg())


def test_resolve_presets_errors_on_chained_no_default():
    """A PresetCfg whose default is another PresetCfg with no 'default'
    must raise ValueError on the inner preset."""

    @configclass
    class InnerNoDefault(PresetCfg):
        option_a: int = 1

    @configclass
    class OuterPreset(PresetCfg):
        default: InnerNoDefault = InnerNoDefault()

    @configclass
    class EnvCfg:
        mode: OuterPreset = OuterPreset()

    with pytest.raises(ValueError, match="no 'default' field"):
        resolve_presets(EnvCfg())


def test_resolve_presets_errors_on_cyclic_preset():
    """Cyclic PresetCfg chain (A.default -> B, B.default -> A) must raise
    ValueError instead of looping forever."""

    @configclass
    class CyclicB(PresetCfg):
        pass

    @configclass
    class CyclicA(PresetCfg):
        default: CyclicB = CyclicB()

    CyclicA.default = CyclicB()
    CyclicB.default = CyclicA()

    @configclass
    class EnvCfg:
        mode: CyclicA = CyclicA()

    with pytest.raises(ValueError, match="[Cc]ycl"):
        resolve_presets(EnvCfg())


def test_resolve_presets_errors_on_cyclic_preset_at_root():
    """Cyclic PresetCfg at root level must raise ValueError, not RecursionError."""

    @configclass
    class RootCyclicB(PresetCfg):
        pass

    @configclass
    class RootCyclicA(PresetCfg):
        default: RootCyclicB = RootCyclicB()

    RootCyclicA.default = RootCyclicB()
    RootCyclicB.default = RootCyclicA()

    with pytest.raises(ValueError, match="[Cc]ycl"):
        resolve_presets(RootCyclicA())
