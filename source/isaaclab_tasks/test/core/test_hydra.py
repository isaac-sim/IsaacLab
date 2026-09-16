# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Self-contained tests for Hydra configuration utilities.

These tests verify the REPLACE-only preset system without depending on
external environment configurations.
"""

import warnings
from dataclasses import dataclass
from typing import Any

import pytest

from isaaclab.utils import config_field, config_to_dict

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


@dataclass
class PhysxCfg:
    backend: str = config_field("physx")
    dt: float = config_field(0.005)
    substeps: int = config_field(2)


@dataclass
class NewtonCfg:
    backend: str = config_field("newton")
    dt: float = config_field(0.002)
    substeps: int = config_field(4)
    solver_iterations: int = config_field(8)


@dataclass
class NoiselessObservationsCfg:
    enable_corruption: bool = config_field(False)
    concatenate_terms: bool = config_field(True)
    noise_scale: float = config_field(0.0)


@dataclass
class FastObservationsCfg:
    enable_corruption: bool = config_field(False)
    concatenate_terms: bool = config_field(False)
    noise_scale: float = config_field(0.0)


@dataclass
class SmallPolicyCfg:
    actor_hidden_dims: list = config_field([64, 32])


@dataclass
class FastPolicyCfg:
    actor_hidden_dims: list = config_field([32, 16])


# =============================================================================
# Composite configs using PresetCfg
# =============================================================================


@dataclass
class SampleEnvCfg:
    decimation: int = config_field(4)
    sim_dt: float = config_field(0.005)


@dataclass
class SampleAgentCfg:
    max_iterations: int = config_field(1000)
    learning_rate: float = config_field(3e-4)


@dataclass
class SimBackendCfg(PresetCfg):
    default: PhysxCfg = config_field(PhysxCfg())
    newton_mjwarp: NewtonCfg = config_field(NewtonCfg())


@dataclass
class ObsModeCfg(PresetCfg):
    default: NoiselessObservationsCfg = config_field(NoiselessObservationsCfg())
    fast: FastObservationsCfg = config_field(FastObservationsCfg())


@dataclass
class PolicyModeCfg(PresetCfg):
    default: SmallPolicyCfg = config_field(SmallPolicyCfg())
    fast: FastPolicyCfg = config_field(FastPolicyCfg())


@dataclass
class PresetCfgEnvCfg:
    decimation: int = config_field(4)
    backend: SimBackendCfg = config_field(SimBackendCfg())
    observations: ObsModeCfg = config_field(ObsModeCfg())


@dataclass
class PresetCfgAgentCfg:
    learning_rate: float = config_field(3e-4)
    policy: PolicyModeCfg = config_field(PolicyModeCfg())


@dataclass
class RootAgentCfg(PresetCfg):
    """Root-level PresetCfg -- the agent config itself is a PresetCfg."""

    default: SampleAgentCfg = config_field(SampleAgentCfg())
    fast: SampleAgentCfg = config_field(SampleAgentCfg(max_iterations=100, learning_rate=1e-3))


# -- Nested PresetCfg-inside-PresetCfg (mirrors scene.base_camera pattern) --


@dataclass
class CameraSmallCfg:
    width: int = config_field(64)
    height: int = config_field(64)


@dataclass
class CameraLargeCfg:
    width: int = config_field(256)
    height: int = config_field(256)


@dataclass
class CameraWideCfg:
    width: int = config_field(512)
    height: int = config_field(128)


@dataclass
class CameraPresetCfg(PresetCfg):
    small: CameraSmallCfg = config_field(CameraSmallCfg())
    large: CameraLargeCfg = config_field(CameraLargeCfg())
    default: CameraSmallCfg = config_field(CameraSmallCfg())


@dataclass
class WideCameraPresetCfg(PresetCfg):
    small: CameraWideCfg = config_field(CameraWideCfg())
    default: CameraWideCfg = config_field(CameraWideCfg())


@dataclass
class BaseSceneCfg:
    num_envs: int = config_field(1024)
    camera: PresetCfg | None = config_field(None)


@dataclass
class ScenePresetCfg(PresetCfg):
    default: BaseSceneCfg = config_field(BaseSceneCfg())
    wide_camera: BaseSceneCfg = config_field(BaseSceneCfg(camera=WideCameraPresetCfg()))
    with_camera: BaseSceneCfg = config_field(BaseSceneCfg(camera=CameraPresetCfg()))


@dataclass
class NestedPresetEnvCfg:
    decimation: int = config_field(4)
    scene: ScenePresetCfg = config_field(ScenePresetCfg())


# -- Scalar PresetCfg and actuator configs (shared by scalar + dict sections) --


@dataclass
class ScalarPresetCfg(PresetCfg):
    default: float = config_field(0.0)
    newton_mjwarp: float = config_field(0.01)


@dataclass
class ActuatorWithPresetCfg:
    joint_names: list = config_field([".*"])
    stiffness: float = config_field(40.0)
    damping: float = config_field(5.0)
    armature: ScalarPresetCfg = config_field(ScalarPresetCfg())


# -- Deep-nested dict configs (event term params pattern) --


@dataclass
class OffsetCfg(PresetCfg):
    """Mimics task-specific offset presets (e.g., AssembledOffsetCfg)."""

    task_a: tuple = config_field((0.0, 0.0, 0.01))
    task_b: tuple = config_field((0.02, 0.0, 0.005))
    default: tuple = config_field(task_a)


@dataclass
class FractionCfg(PresetCfg):
    task_a: tuple = config_field((0.05, 0.5))
    task_b: tuple = config_field((0.3, 1.0))
    default: tuple = config_field(task_a)


@dataclass
class JointNamesCfg(PresetCfg):
    default: list[str] | None = config_field(None)
    robot_a: list[str] = config_field(None)
    robot_b: list[str] = config_field(None)


@dataclass
class EntityCfg:
    """Mimics SceneEntityCfg with a preset-valued field."""

    name: str = config_field("robot")
    joint_names: list[str] | None = config_field(None)


@dataclass
class InnerTermCfg:
    """Mimics an EventTermCfg with params containing presets."""

    func: str = config_field("reset_fn")
    params: dict = config_field(None)

    def __post_init__(self):
        if self.params is None:
            self.params = {
                "offset": OffsetCfg(),
                "fraction": FractionCfg(),
                "robot_cfg": EntityCfg(name="robot", joint_names=JointNamesCfg()),
            }


@dataclass
class OuterTermCfg:
    """Mimics a chained reset term with nested terms dict."""

    func: str = config_field("chain_fn")
    params: dict = config_field(None)

    def __post_init__(self):
        if self.params is None:
            self.params = {
                "terms": {
                    "step_one": InnerTermCfg(),
                }
            }


@dataclass
class DeepDictEnvCfg:
    decimation: int = config_field(4)
    events: OuterTermCfg = config_field(OuterTermCfg())


@dataclass
class DictPresetTermCfg:
    """Outer term where the terms dict is itself a preset (resolves to a dict)."""

    func: str = config_field("term_choice")
    params: dict = config_field(None)

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


@dataclass
class PresetResolvesToDictEnvCfg:
    decimation: int = config_field(4)
    events: DictPresetTermCfg = config_field(DictPresetTermCfg())


# =============================================================================
# Helpers
# =============================================================================


def _apply(env_cfg, agent_cfg=None, global_presets=None, preset_sel=None, preset_scalar=None):
    """Collect presets, resolve defaults, build hydra dict, and apply overrides."""
    if agent_cfg is None:
        agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    hydra_cfg = {"env": config_to_dict(env_cfg), "agent": config_to_dict(agent_cfg)}
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

    @dataclass
    class _SolverPresetsCfg(PresetCfg):
        default: PhysxCfg = config_field(PhysxCfg())
        newton_kamino: NewtonCfg = config_field(NewtonCfg())

    cfg = _SolverPresetsCfg()
    with pytest.warns(FutureWarning, match="Preset 'kamino' is deprecated"):
        assert cfg.kamino is cfg.newton_kamino


@pytest.mark.parametrize(
    "legacy_name,canonical_name",
    [
        ("ovrtx_renderer", "ovrtx"),
        ("isaacsim_rtx_renderer", "isaacsim_rtx"),
    ],
)
def test_legacy_renderer_suffix_attribute_alias_warns(legacy_name, canonical_name):
    """The suffixed renderer preset names alias to their ``_renderer``-less fields during deprecation."""

    @dataclass
    class _RendererPresetsCfg(PresetCfg):
        default: PhysxCfg = config_field(PhysxCfg())
        ovrtx: PhysxCfg = config_field(PhysxCfg())
        isaacsim_rtx: PhysxCfg = config_field(PhysxCfg())

    cfg = _RendererPresetsCfg()
    with pytest.warns(FutureWarning, match=f"Preset '{legacy_name}' is deprecated"):
        assert getattr(cfg, legacy_name) is getattr(cfg, canonical_name)


def test_legacy_alias_suppressed_when_legacy_name_is_real_field():
    """An env that legitimately defines ``newton`` should not warn or be remapped."""

    @dataclass
    class _ShadowingCfg(PresetCfg):
        default: PhysxCfg = config_field(PhysxCfg())
        newton: PhysxCfg = config_field(PhysxCfg())
        newton_mjwarp: NewtonCfg = config_field(NewtonCfg())

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
    hydra_cfg = {"env": config_to_dict(env_cfg), "agent": config_to_dict(agent_cfg)}
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], [], [], presets)
    assert isinstance(env_cfg.backend, PhysxCfg)
    assert isinstance(env_cfg.observations, NoiselessObservationsCfg)
    assert isinstance(agent_cfg.policy, SmallPolicyCfg)


def test_presetcfg_cli_selection(class_presets):
    """Path selection replaces with chosen preset."""
    env_cfg, agent_cfg, presets = class_presets
    hydra_cfg = {"env": config_to_dict(env_cfg), "agent": config_to_dict(agent_cfg)}
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], [("env", "backend", "newton_mjwarp")], [], presets)
    assert isinstance(env_cfg.backend, NewtonCfg)
    assert env_cfg.backend.dt == 0.002


def test_presetcfg_global_broadcast(class_presets):
    """Global preset 'fast' broadcasts across env and agent PresetCfg fields."""
    env_cfg, agent_cfg, presets = class_presets
    hydra_cfg = {"env": config_to_dict(env_cfg), "agent": config_to_dict(agent_cfg)}
    apply_overrides(env_cfg, agent_cfg, hydra_cfg, ["fast"], [], [], presets)
    assert isinstance(env_cfg.observations, FastObservationsCfg)
    assert isinstance(agent_cfg.policy, FastPolicyCfg)


def test_presetcfg_path_selection_others_default(class_presets):
    """Path preset on one field, others get auto-default."""
    env_cfg, agent_cfg, presets = class_presets
    hydra_cfg = {"env": config_to_dict(env_cfg), "agent": config_to_dict(agent_cfg)}
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
    assert set(presets["scene"].keys()) == {"default", "wide_camera", "with_camera"}
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


def test_nested_presetcfg_global_preset_uses_selected_parent_branch():
    """Same nested preset names should resolve inside the selected parent branch."""
    env_cfg, _ = _apply(NestedPresetEnvCfg(), global_presets=["wide_camera", "small"])

    assert isinstance(env_cfg.scene, BaseSceneCfg)
    assert isinstance(env_cfg.scene.camera, CameraWideCfg)


def test_nested_presetcfg_path_preset_uses_selected_parent_branch():
    """Unqualified public paths should still resolve against the selected active branch."""
    sel = [("env", "scene", "wide_camera"), ("env", "scene.camera", "small")]
    env_cfg, _ = _apply(NestedPresetEnvCfg(), preset_sel=sel)

    assert isinstance(env_cfg.scene, BaseSceneCfg)
    assert isinstance(env_cfg.scene.camera, CameraWideCfg)


# =============================================================================
# Tests: root-level PresetCfg with nested PresetCfg inside alternatives
# (mirrors CartpoleCameraEnvCfg structure)
# =============================================================================


@dataclass
class RendererACfg:
    backend: str = config_field("rtx")


@dataclass
class RendererBCfg:
    backend: str = config_field("warp")


@dataclass
class RendererPresetCfg(PresetCfg):
    default: RendererACfg = config_field(RendererACfg())
    newton_renderer: RendererBCfg = config_field(RendererBCfg())


@dataclass
class SensorBaseCfg:
    data_types: list[str] = config_field([])
    width: int = config_field(100)
    height: int = config_field(100)
    renderer: RendererPresetCfg = config_field(RendererPresetCfg())


@dataclass
class SensorPresetCfg(PresetCfg):
    default: SensorBaseCfg = config_field(SensorBaseCfg(data_types=["rgb"]))
    depth: SensorBaseCfg = config_field(SensorBaseCfg(data_types=["depth"]))


@dataclass
class RootEnvBaseCfg:
    decimation: int = config_field(2)
    sensor: SensorPresetCfg = config_field(SensorPresetCfg())
    obs_shape: list[int] = config_field([100, 100, 3])


@dataclass
class RootPresetEnvCfg(PresetCfg):
    default: RootEnvBaseCfg = config_field(RootEnvBaseCfg())
    depth: RootEnvBaseCfg = config_field(RootEnvBaseCfg(obs_shape=[100, 100, 1]))


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


@dataclass
class OptionalFeatureCfg:
    buffer_size: int = config_field(200)
    export_path: str = config_field(".")


@dataclass
class OptionalFeaturePresetCfg(PresetCfg):
    default: Any = config_field(None)
    enabled: OptionalFeatureCfg = config_field(OptionalFeatureCfg())


@dataclass
class EnvWithOptionalFeatureCfg:
    decimation: int = config_field(4)
    optional_feature: OptionalFeaturePresetCfg = config_field(OptionalFeaturePresetCfg())


def test_presetcfg_none_default_auto_applies():
    """PresetCfg with default=None auto-applies None without crashing."""
    env_cfg, _ = _apply(EnvWithOptionalFeatureCfg())
    assert env_cfg.optional_feature is None


def test_presetcfg_none_default_cli_selects_enabled():
    """PresetCfg with default=None can be overridden to a real config via CLI."""
    env_cfg = EnvWithOptionalFeatureCfg()
    agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    hydra_cfg = {"env": config_to_dict(env_cfg), "agent": config_to_dict(agent_cfg)}
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


@dataclass
class ScalarPresetEnvCfg:
    decimation: int = config_field(4)
    actuator: ActuatorWithPresetCfg = config_field(ActuatorWithPresetCfg())


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


@dataclass
class RobotCfg:
    prim_path: str = config_field("/World/Robot")
    actuators: dict = config_field(None)

    def __post_init__(self):
        if self.actuators is None:
            self.actuators = {"legs": ActuatorWithPresetCfg()}


@dataclass
class DictPresetEnvCfg:
    decimation: int = config_field(4)
    robot: RobotCfg = config_field(RobotCfg())


def test_collect_presets_traverses_dict_values():
    """collect_presets finds PresetCfg inside dict-held configuration dataclasses."""
    cfg = DictPresetEnvCfg()
    presets = collect_presets(cfg)
    assert "robot.actuators.legs.armature" in presets
    assert presets["robot.actuators.legs.armature"]["default"] == 0.0
    assert presets["robot.actuators.legs.armature"]["newton_mjwarp"] == 0.01


def test_resolve_presets_traverses_dict_values():
    """resolve_presets resolves PresetCfg inside dict-held configuration dataclasses."""
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
    """preset() factory works inside dict-held configuration dataclasses."""

    @dataclass
    class ActuatorCfgFactory:
        joint_names: list = config_field([".*"])
        armature: object = config_field(None)

        def __post_init__(self):
            if self.armature is None:
                self.armature = preset(default=0.0, newton_mjwarp=0.01, physx=0.0)

    @dataclass
    class RobotCfgFactory:
        actuators: dict = config_field(None)

        def __post_init__(self):
            if self.actuators is None:
                self.actuators = {"legs": ActuatorCfgFactory()}

    @dataclass
    class EnvCfgFactory:
        robot: RobotCfgFactory = config_field(RobotCfgFactory())

    cfg = EnvCfgFactory()
    presets = collect_presets(cfg)
    assert "robot.actuators.legs.armature" in presets
    assert presets["robot.actuators.legs.armature"]["default"] == 0.0
    assert presets["robot.actuators.legs.armature"]["newton_mjwarp"] == 0.01
    assert presets["robot.actuators.legs.armature"]["physx"] == 0.0


# =============================================================================
# Tests: rough terrain config regressions
# =============================================================================


def test_go2_rough_legacy_newton_alias_resolves_to_newton_mjwarp():
    """Real-config alias path: ``presets=newton`` against an actual env cfg resolves to newton_mjwarp."""
    from isaaclab_newton.physics import MJWarpSolverCfg

    from isaaclab_tasks.core.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg

    with pytest.warns(FutureWarning, match="Preset 'newton' is deprecated"):
        env_cfg, _ = _apply(UnitreeGo2RoughEnvCfg(), global_presets=["newton"])
    assert isinstance(env_cfg.sim.physics.solver_cfg, MJWarpSolverCfg)


def test_velocity_events_newton_mjwarp_keeps_base_com_randomization():
    """MJWarp velocity configs should retain base center-of-mass randomization."""
    from isaaclab_tasks.core.velocity import mdp
    from isaaclab_tasks.core.velocity.velocity_env_cfg import EventsCfg

    events = resolve_presets(EventsCfg(), {"newton_mjwarp"})

    assert events.base_com is not None
    assert events.base_com.func is mdp.randomize_rigid_body_com
    assert events.base_com.mode == "startup"
    assert events.base_com.params["com_range"] == {
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
    }


# =============================================================================
# Tests: PresetCfg inside deeply nested dicts (e.g., event term params)
# =============================================================================


def test_collect_presets_deep_nested_dicts():
    """collect_presets discovers PresetCfg inside nested dict and configuration dataclass chains."""
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

    @dataclass
    class BodyNameCfg(PresetCfg):
        default: str = config_field("generic_body")

    @dataclass
    class TermWithBody:
        func: str = config_field("some_fn")
        params: dict = config_field(None)

        def __post_init__(self):
            if self.params is None:
                self.params = {"cfg": EntityCfg(name="robot", joint_names=BodyNameCfg())}

    @dataclass
    class EnvWithBody:
        events: TermWithBody = config_field(TermWithBody())

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

    @dataclass
    class FactoryEnvCfg:
        damping: object = config_field(None)

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

    @dataclass
    class MutablePresetCfg(PresetCfg):
        default: str = config_field("original_default")
        alt: str = config_field("alternative")

    instance = MutablePresetCfg()
    assert instance.default == "original_default"

    MutablePresetCfg.default = "robot_specific_default"

    presets = collect_presets(instance)
    assert "" in presets
    assert presets[""]["default"] == "robot_specific_default"

    MutablePresetCfg.default = "original_default"


def test_collect_fields_includes_dynamic_class_attrs():
    """Fields added to PresetCfg class at runtime are discovered."""

    @dataclass
    class ExtensiblePresetCfg(PresetCfg):
        default: str = config_field("base")
        alt_a: str = config_field("a")

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
    hydra_cfg = {"env": config_to_dict(env_cfg), "agent": config_to_dict(agent_cfg)}
    with pytest.raises(ValueError, match="Unknown or inactive preset group"):
        apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], [("env", "nonexistent", "val")], [], presets)


def test_apply_overrides_unknown_preset_name_raises():
    """apply_overrides raises ValueError for unknown preset name."""
    env_cfg = PresetCfgEnvCfg()
    agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    hydra_cfg = {"env": config_to_dict(env_cfg), "agent": config_to_dict(agent_cfg)}
    with pytest.raises(ValueError, match="Unknown preset 'nonexistent'"):
        apply_overrides(env_cfg, agent_cfg, hydra_cfg, [], [("env", "backend", "nonexistent")], [], presets)


def test_apply_overrides_conflicting_globals_raises():
    """Two global presets matching the same path cause ValueError."""

    @dataclass
    class TwoAltsPresetCfg(PresetCfg):
        default: str = config_field("d")
        opt_a: str = config_field("a")
        opt_b: str = config_field("b")

    @dataclass
    class ConflictEnvCfg:
        mode: TwoAltsPresetCfg = config_field(TwoAltsPresetCfg())

    env_cfg = ConflictEnvCfg()
    agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    hydra_cfg = {"env": config_to_dict(env_cfg), "agent": config_to_dict(agent_cfg)}
    with pytest.raises(ValueError, match="Conflicting global presets"):
        apply_overrides(env_cfg, agent_cfg, hydra_cfg, ["opt_a", "opt_b"], [], [], presets)


def test_apply_overrides_aliased_globals_no_conflict():
    """Two global presets resolving to equal values do not raise.

    Mirrors the Lift ObjectCfg pattern where ``newton_mjwarp = cube`` creates
    separate but equal dataclass instances after config_field processing.
    """

    @dataclass
    class SharedCfg:
        value: int = config_field(42)

    cube_val = SharedCfg()
    mjwarp_val = SharedCfg()

    @dataclass
    class AliasedPresetCfg(PresetCfg):
        default: str = config_field("d")
        cube: SharedCfg = config_field(cube_val)
        newton_mjwarp: SharedCfg = config_field(mjwarp_val)

    @dataclass
    class AliasedEnvCfg:
        mode: AliasedPresetCfg = config_field(AliasedPresetCfg())

    env_cfg = AliasedEnvCfg()
    agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    assert presets["env"]["mode"]["cube"] is not presets["env"]["mode"]["newton_mjwarp"]
    assert presets["env"]["mode"]["cube"] == presets["env"]["mode"]["newton_mjwarp"]
    hydra_cfg = {"env": config_to_dict(env_cfg), "agent": config_to_dict(agent_cfg)}
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
    hydra_cfg = {"env": config_to_dict(env_cfg), "agent": config_to_dict(agent_cfg)}
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
    hydra_cfg = {"env": config_to_dict(env_cfg), "agent": config_to_dict(agent_cfg)}
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
    hydra_cfg = {"env": config_to_dict(env_cfg), "agent": config_to_dict(agent_cfg)}
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


def test_scalar_override_kamino_solver_config():
    """Concrete Kamino solver fields can be overridden through Hydra scalar paths."""
    from isaaclab_newton.physics import KaminoPADMMSolverCfg, NewtonCfg

    @dataclass
    class KaminoPhysicsPreset(PresetCfg):
        default: NewtonCfg = config_field(NewtonCfg())
        newton_kamino: NewtonCfg = config_field(NewtonCfg(solver_cfg=KaminoPADMMSolverCfg(sparse_jacobian=True)))

    @dataclass
    class KaminoEnvCfg:
        physics: KaminoPhysicsPreset = config_field(KaminoPhysicsPreset())

    env_cfg = KaminoEnvCfg()
    agent_cfg = PresetCfgAgentCfg()
    presets = {"env": collect_presets(env_cfg), "agent": collect_presets(agent_cfg)}
    hydra_cfg = {"env": config_to_dict(env_cfg), "agent": config_to_dict(agent_cfg)}
    apply_overrides(
        env_cfg,
        agent_cfg,
        hydra_cfg,
        ["newton_kamino"],
        [],
        [("env.physics.solver_cfg.dynamics_solver_cfg.max_iterations", "25")],
        presets,
    )
    assert isinstance(env_cfg.physics.solver_cfg, KaminoPADMMSolverCfg)
    assert env_cfg.physics.solver_cfg.dynamics_solver_cfg.max_iterations == 25


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

    @dataclass
    class NoDefaultPreset(PresetCfg):
        option_a: int = config_field(1)

    @dataclass
    class EnvCfg:
        mode: NoDefaultPreset = config_field(NoDefaultPreset())

    with pytest.raises(ValueError, match="no 'default' field"):
        resolve_presets(EnvCfg())


def test_resolve_presets_errors_on_chained_no_default():
    """A PresetCfg whose default is another PresetCfg with no 'default'
    must raise ValueError on the inner preset."""

    @dataclass
    class InnerNoDefault(PresetCfg):
        option_a: int = config_field(1)

    @dataclass
    class OuterPreset(PresetCfg):
        default: InnerNoDefault = config_field(InnerNoDefault())

    @dataclass
    class EnvCfg:
        mode: OuterPreset = config_field(OuterPreset())

    with pytest.raises(ValueError, match="no 'default' field"):
        resolve_presets(EnvCfg())


def test_resolve_presets_errors_on_cyclic_preset():
    """Cyclic PresetCfg chain (A.default -> B, B.default -> A) must raise
    ValueError instead of looping forever."""

    @dataclass
    class CyclicB(PresetCfg):
        pass

    @dataclass
    class CyclicA(PresetCfg):
        default: CyclicB = config_field(CyclicB())

    CyclicA.default = CyclicB()
    CyclicB.default = CyclicA()

    @dataclass
    class EnvCfg:
        mode: CyclicA = config_field(CyclicA())

    with pytest.raises(ValueError, match="[Cc]ycl"):
        resolve_presets(EnvCfg())


def test_resolve_presets_errors_on_cyclic_preset_at_root():
    """Cyclic PresetCfg at root level must raise ValueError, not RecursionError."""

    @dataclass
    class RootCyclicB(PresetCfg):
        pass

    @dataclass
    class RootCyclicA(PresetCfg):
        default: RootCyclicB = config_field(RootCyclicB())

    RootCyclicA.default = RootCyclicB()
    RootCyclicB.default = RootCyclicA()

    with pytest.raises(ValueError, match="[Cc]ycl"):
        resolve_presets(RootCyclicA())


# =============================================================================
# Tests: typed-selector validation (physics=/renderer= must hit their type)
# =============================================================================

from isaaclab.physics import PhysicsCfg as _RealPhysicsCfg  # noqa: E402

from isaaclab_tasks.utils.preset_target import PresetTarget  # noqa: E402


@dataclass
class _NewtonPhysicsCfg(_RealPhysicsCfg):
    """Minimal real ``PhysicsCfg`` subclass so isinstance bucketing routes to PHYSICS."""

    dt: float = config_field(0.002)


@dataclass
class _PhysxPhysicsCfg(_RealPhysicsCfg):
    dt: float = config_field(0.005)


def test_validate_typed_presets_passes_when_selector_hits_its_type():
    """``physics=newton_mjwarp`` that landed on a PhysicsCfg does not raise."""
    hydra_mod._validate_typed_presets(
        {PresetTarget.PHYSICS: {"newton_mjwarp"}},
        typed_hits={"newton_mjwarp": {PresetTarget.PHYSICS}},
    )


def test_validate_typed_presets_raises_when_selector_misses_its_type():
    """``physics=newton_mjwarp`` that never landed on a PhysicsCfg must raise."""
    with pytest.raises(ValueError, match="physics=newton_mjwarp"):
        hydra_mod._validate_typed_presets({PresetTarget.PHYSICS: {"newton_mjwarp"}}, typed_hits={})


def test_validate_typed_presets_ignores_broadcast_presets():
    """A plain ``presets=`` broadcast is never in ``requested``, so it is trusted."""
    # No typed selectors requested -> nothing to validate, even with no hits.
    hydra_mod._validate_typed_presets({}, typed_hits={})


def test_resolve_active_presets_records_physics_hit_for_selector():
    """End-to-end: selecting a name that resolves to a real PhysicsCfg records a PHYSICS hit."""

    @dataclass
    class PhysicsPresetCfg(PresetCfg):
        default: _PhysxPhysicsCfg = config_field(_PhysxPhysicsCfg())
        newton_mjwarp: _NewtonPhysicsCfg = config_field(_NewtonPhysicsCfg())

    @dataclass
    class EnvWithPhysicsCfg:
        physics: PhysicsPresetCfg = config_field(PhysicsPresetCfg())

    typed_hits: dict[str, set[PresetTarget]] = {}
    hydra_mod._resolve_active_presets(
        EnvWithPhysicsCfg(), ["newton_mjwarp"], {}, root_path="env", typed_hits=typed_hits
    )
    assert PresetTarget.PHYSICS in typed_hits.get("newton_mjwarp", set())
    # physics=newton_mjwarp therefore validates.
    hydra_mod._validate_typed_presets({PresetTarget.PHYSICS: {"newton_mjwarp"}}, typed_hits)


def test_resolve_active_presets_no_physics_hit_for_scalar_preset():
    """A name resolving only to a scalar records no typed hit, so a physics= selector raises."""

    @dataclass
    class EnvWithScalarOnlyCfg:
        # ``newton_mjwarp`` here only tunes a scalar -- no PhysicsCfg involved.
        armature: PresetCfg = config_field(preset(default=0.0, newton_mjwarp=0.01))

    consumed: set[str] = set()
    typed_hits: dict[str, set[PresetTarget]] = {}
    hydra_mod._resolve_active_presets(
        EnvWithScalarOnlyCfg(),
        ["newton_mjwarp"],
        {},
        root_path="env",
        consumed_selected=consumed,
        typed_hits=typed_hits,
    )
    assert "newton_mjwarp" in consumed and PresetTarget.PHYSICS not in typed_hits.get("newton_mjwarp", set())
    # presets=newton_mjwarp (broadcast) is trusted: no entry in ``requested`` -> no error.
    hydra_mod._validate_typed_presets({}, typed_hits)
    # physics=newton_mjwarp (typed selector) must error.
    with pytest.raises(ValueError, match="physics=newton_mjwarp"):
        hydra_mod._validate_typed_presets({PresetTarget.PHYSICS: {"newton_mjwarp"}}, typed_hits)


# =============================================================================
# Tests: play-mode overrides
# =============================================================================


def test_register_task_play_mode_applies_play_mode(monkeypatch):
    """``register_task(play_mode=True)`` applies the env cfg's play-mode overrides after loading."""
    import sys

    import gymnasium as gym

    @dataclass
    class PlayModeEnvCfg:
        played: bool = config_field(False)

        def play_mode(self):
            self.played = True

    gym.register(
        id="Isaac-Hydra-PlayMode-Test",
        entry_point="dummy:Env",
        kwargs={"env_cfg_entry_point": PlayModeEnvCfg},
    )
    monkeypatch.setattr(sys, "argv", ["test"])
    try:
        env_cfg, _, _ = hydra_mod.register_task("Isaac-Hydra-PlayMode-Test", None, play_mode=True)
        assert env_cfg.played
        env_cfg, _, _ = hydra_mod.register_task("Isaac-Hydra-PlayMode-Test", None)
        assert not env_cfg.played
    finally:
        del gym.registry["Isaac-Hydra-PlayMode-Test"]
