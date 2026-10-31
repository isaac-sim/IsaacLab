# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CLI spelling, defaults, and isolation from launcher arguments.

Preset-name validation and task composition are covered in ``test_hydra.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="train.py", add_help=False)
    parser.add_argument("--task", type=str, default=None)
    return parser


# ---------------------------------------------------------------------------
# setup_preset_cli: normalize task selectors without changing launcher settings
# ---------------------------------------------------------------------------


def test_setup_preset_cli_returns_remainder_only(monkeypatch):
    """Without any preset tokens, the remainder is just the un-touched
    non-argparse tokens (Hydra path overrides, etc.)."""
    original = ["train.py", "--task=Foo-v0", "env.sim.dt=0.001"]
    monkeypatch.setattr("sys.argv", original)
    from isaaclab_tasks.utils.hydra import setup_preset_cli

    args, remaining = setup_preset_cli(_make_parser())
    assert args.task == "Foo-v0"
    assert remaining == ["env.sim.dt=0.001"]
    # setup_preset_cli must NOT mutate sys.argv -- the caller controls when to assign.
    assert sys.argv == original


def test_setup_preset_cli_passes_typed_tokens_verbatim(monkeypatch):
    """Preset tokens come back in their original ``physics=`` / ``renderer=`` /
    ``presets=`` form so hydra can parse them directly and callers can intersect
    with callback returns in matching vocabulary."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--task=Foo-v0",
            "physics=newton_mjwarp",
            "renderer=newton_renderer",
            "presets=albedo,depth",
            "env.sim.dt=0.001",
        ],
    )
    from isaaclab_tasks.utils.hydra import setup_preset_cli

    _, remaining = setup_preset_cli(_make_parser())
    assert remaining == [
        "physics=newton_mjwarp",
        "renderer=newton_renderer",
        "presets=albedo,depth",
        "env.sim.dt=0.001",
    ]


@pytest.mark.parametrize("spelling", ["hydra", "flags", "equals"])
def test_setup_preset_cli_namespace_carries_no_preset_attributes(monkeypatch, spelling):
    """Task selectors never leak into Kit arguments; explicit Kit settings stay untouched."""
    selections = dict(physics="newton_mjwarp", renderer="newton_renderer", visualizer="newton_gl")
    tokens = []
    for key, value in selections.items():
        if spelling == "flags":
            tokens.extend((f"--{key}", value))
        else:
            tokens.append(f"{key}={value}" if spelling == "hydra" else f"--{key}={value}")
    kit_args = "--renderer=PathTracing --/rtx/pathtracing/spp=16"
    original = ["train.py", "--task=Foo-v0", *tokens, "presets=albedo", "--kit_args", kit_args]
    monkeypatch.setattr("sys.argv", original)
    from isaaclab_tasks.utils.hydra import setup_preset_cli

    parser = _make_parser()
    parser.add_argument("--kit_args")
    parser.set_defaults(physics="isaacsim_physx", renderer="isaacsim_rtx", visualizer=["kit"])
    args, remaining = setup_preset_cli(parser)
    assert remaining == [*(f"{key}={value}" for key, value in selections.items()), "presets=albedo"]
    assert vars(args) == dict(task="Foo-v0", kit_args=kit_args)
    assert sys.argv == original
    _, defaults = setup_preset_cli(parser, [])
    assert defaults == ["physics=isaacsim_physx", "renderer=isaacsim_rtx", "visualizer=kit"]
    with pytest.raises(SystemExit):
        setup_preset_cli(parser, [*tokens, "renderer=ovrtx"])
    for key in selections:
        with pytest.raises(SystemExit):
            setup_preset_cli(parser, [f"--{key}"])


def test_enumerate_task_presets_routes_by_target_match(monkeypatch):
    """Variants bucket through :meth:`PresetTarget.matches`.

    PhysicsCfg subclass instances route to PHYSICS, RendererCfg subclass
    instances route to RENDERER, PhysicsCfg-containing SimulationCfg bundles
    route to PHYSICS, and values matching no target fall into DOMAIN.
    """
    from isaaclab.physics import PhysicsCfg
    from isaaclab.renderers.renderer_cfg import RendererCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.utils import configclass

    from isaaclab_tasks.utils.hydra import PresetTarget, enumerate_task_presets, preset

    @configclass
    class _PhysVariant(PhysicsCfg):
        class_type: str = "mock"

    @configclass
    class _PhysWrapper(PhysicsCfg):
        # Mirrors NewtonCfg's "wrapper holds an inner solver" shape: still
        # subclasses PhysicsCfg, so the base-class isinstance check still
        # buckets it correctly regardless of any nested member type.
        class_type: str = "mock_wrapper"
        inner: object = None

    @configclass
    class _RendVariant(RendererCfg):
        pass

    walked = {
        "physics": {
            "default": _PhysVariant(),
            "physx": _PhysVariant(),
            "newton_mjwarp": _PhysWrapper(inner=_PhysVariant()),
            "newton_kamino": _PhysWrapper(inner=_PhysVariant()),
        },
        "renderer": {
            "default": _RendVariant(),
            "newton_renderer": _RendVariant(),
        },
        "sim": {
            "default": SimulationCfg(dt=1 / 60, physics=_PhysVariant()),
            "simulation_physx": SimulationCfg(dt=1 / 60, physics=_PhysVariant()),
        },
        "sim_no_backend": {
            "default": SimulationCfg(dt=1 / 240),
            "plain": SimulationCfg(dt=1 / 240),
        },
        "weight": {  # cfgs whose type is not a typed-target base subclass -> DOMAIN
            "default": 1.0,
            "light": 0.5,
            "heavy": 2.0,
        },
    }
    monkeypatch.setattr(
        "isaaclab_tasks.utils.parse_cfg.load_cfg_from_registry",
        lambda *_: {path: preset(**fields) for path, fields in walked.items()},
    )
    result = enumerate_task_presets("Test")
    # All physics variants bucket to PHYSICS (including the wrapper-shaped ones).
    assert {"physx", "newton_mjwarp", "newton_kamino", "simulation_physx"} <= set(result[PresetTarget.PHYSICS])
    assert "newton_renderer" in result[PresetTarget.RENDERER]
    # Primitive-typed variants land in DOMAIN.
    assert {"plain", "light", "heavy"} <= set(result[PresetTarget.DOMAIN])
    # 'default' is filtered out everywhere -- it's the fallback, not a selectable name.
    for bucket in result.values():
        assert "default" not in bucket


# ---------------------------------------------------------------------------
# --help: section description renders the variant listing
# ---------------------------------------------------------------------------


def test_help_without_task_says_pass_task(monkeypatch, capsys):
    """``--help`` without ``--task`` tells the user to pass ``--task=X``,
    once on the section description rather than repeated per-flag.
    """
    monkeypatch.setattr("sys.argv", ["train.py", "--help"])
    from isaaclab_tasks.utils import hydra, setup_preset_cli

    # Task preset parsing, categories, and resolution share one module.
    assert not any(Path(hydra.__file__).with_name(module).exists() for module in ("preset_cli.py", "preset_target.py"))
    parser = argparse.ArgumentParser(prog="train.py")  # default add_help=True
    parser.add_argument("--task", type=str, default=None)
    with pytest.raises(SystemExit):
        setup_preset_cli(parser)
    out = capsys.readouterr().out
    assert out.count("Pass `--task=X`") == 1


@pytest.mark.parametrize(
    "build_key, expected_phrases",
    [
        pytest.param(
            "empty",
            [
                "physics=NAME (typed) selects a physics backend. Available: (none)",
                "renderer=NAME (typed) selects a renderer backend. Available: (none)",
                "presets=NAME[,NAME,...] broadcast: applied to every matching PresetCfg. Available: (none)",
            ],
            id="zero_variants_everywhere",
        ),
        pytest.param(
            "mixed",
            [
                "physics=NAME (typed) selects a physics backend. Available: - my_phys",
                "renderer=NAME (typed) selects a renderer backend. Available: - my_rend",
                "presets=NAME[,NAME,...] broadcast: applied to every matching PresetCfg. Available: - heavy - light",
            ],
            id="all_three_buckets_populated",
        ),
    ],
)
def test_help_text_branch_strings(monkeypatch, capsys, build_key, expected_phrases):
    """Help uses the supplied argv's last task and lists variants in their typed sections."""
    from isaaclab.physics import PhysicsCfg
    from isaaclab.renderers.renderer_cfg import RendererCfg
    from isaaclab.utils import configclass

    from isaaclab_tasks.utils.hydra import preset

    @configclass
    class _HelpPhysCfg(PhysicsCfg):
        class_type: str = "mock"

    @configclass
    class _HelpRendCfg(RendererCfg):
        pass

    @configclass
    class _EmptyCfg:
        pass

    @configclass
    class _MixedCfg:
        physics: object = preset(default=_HelpPhysCfg(), my_phys=_HelpPhysCfg())
        renderer: object = preset(default=_HelpRendCfg(), my_rend=_HelpRendCfg())
        weight: object = preset(default=1.0, light=0.5, heavy=2.0)

    builders = {
        "empty": _EmptyCfg,
        "mixed": _MixedCfg,
    }

    import isaaclab_tasks.utils.parse_cfg as parse_cfg

    def load(task, entry):
        assert task == "Fake-v0"
        return builders[build_key]()

    monkeypatch.setattr(parse_cfg, "load_cfg_from_registry", load)
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Unrelated-v0", "--help"])
    argv = (
        ["--task=Old-v0", "--task", "Fake-v0", "-h"]
        if build_key == "empty"
        else ["--task", "Old-v0", "--task=Fake-v0", "--help"]
    )
    from isaaclab_tasks.utils.hydra import setup_preset_cli

    parser = argparse.ArgumentParser(prog="train.py")
    parser.add_argument("--task", type=str, default=None)
    with pytest.raises(SystemExit):
        setup_preset_cli(parser, argv)
    # Collapse argparse line-wrapping so substring checks survive width changes.
    flat = " ".join(capsys.readouterr().out.split())

    for phrase in expected_phrases:
        assert phrase in flat, f"Missing phrase: {phrase!r}"


@pytest.mark.parametrize(
    "task_name",
    [
        "IsaacContrib-PickPlace-GR1T2-Abs",
        "IsaacContrib-PickPlace-GR1T2-WaistEnabled-Abs",
        "IsaacContrib-PickPlace-Locomanipulation-G1-Abs",
        "IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs",
        "IsaacContrib-Stack-Cube-SO101-IK-Abs-v0",
    ],
)
@pytest.mark.parametrize("physics", [None, "isaacsim_physx", "ovphysx", "newton_mjwarp"])
def test_teleop_tasks_resolve_public_physics_selectors(monkeypatch, task_name, physics):
    """Registered construction prepares the selected backend before simulation starts."""
    import gymnasium as gym
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_ov.physics import OvPhysxCfg
    from isaaclab_physx.physics import PhysxCfg

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils import instantiate

    import isaaclab_tasks  # noqa: F401 -- registers the public task IDs
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.hydra import setup_preset_cli

    arguments = ["--task", task_name]
    if physics is not None:
        arguments.extend(["--physics", physics])
    _, selections = setup_preset_cli(_make_parser(), arguments)
    cfg = parse_env_cfg(task_name, device="cpu", num_envs=1, overrides=selections)
    expected_type = {None: PhysxCfg, "isaacsim_physx": PhysxCfg, "ovphysx": OvPhysxCfg, "newton_mjwarp": NewtonCfg}
    assert isinstance(cfg.sim.physics, expected_type[physics])
    assert cfg.scene.num_envs == 1 and cfg.sim.device == "cpu"
    if "SO101" in task_name:
        assert cfg.scene.robot.spawn.variants["Physics"] == ("physics" if physics == "newton_mjwarp" else "physx")
    if physics == "newton_mjwarp":
        assert isinstance(cfg.sim.physics.solver_cfg, MJWarpSolverCfg)
        if "SO101" in task_name:
            assert cfg.sim.physics.solver_cfg.iterations == 100
            assert cfg.sim.physics.solver_cfg.impratio == 10.0
            assert cfg.sim.physics.solver_cfg.nconmax == 600

    class ConstructionReached(Exception):
        pass

    from isaaclab_tasks.contrib.packing_table import PackingTableUsdFileCfg
    from isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_env_cfg import GR1T2KitlessUsdFileCfg

    original_action_type = getattr(getattr(cfg.actions, "upper_body_ik", None), "class_type", None)

    def inspect_before_simulation(self, cfg, *args, **kwargs):
        self._is_closed = True
        standalone = physics in ("ovphysx", "newton_mjwarp")
        if "GR1T2" in task_name:
            action = cfg.actions.upper_body_ik
            assert action.kitless_target_conditioning_enabled is standalone
            if standalone:
                assert Path(action.controller.urdf_path).is_file()
                assert isinstance(cfg.scene.robot.spawn, GR1T2KitlessUsdFileCfg)
            else:
                assert action.controller.urdf_path is None
            expected_waist = "WaistEnabled" in task_name
            assert ("waist_yaw_joint" in action.pink_controlled_joint_names) is expected_waist
        if "SO101" not in task_name:
            assert isinstance(cfg.scene.packing_table.spawn, PackingTableUsdFileCfg) is standalone
            if physics == "newton_mjwarp":
                if "Locomanipulation" in task_name:
                    assert cfg.sim.physics.solver_cfg.integrator == "implicitfast"
                    assert cfg.sim.physics.num_substeps == 2
                else:
                    assert cfg.sim.physics.solver_cfg.nconmax == 4096
                    assert cfg.sim.physics.solver_cfg.njmax == 1536
                if "FixedBaseUpperBodyIK" in task_name:
                    from isaaclab_tasks.contrib.locomanip_pick_place.g1_kitless_newton import (
                        G1NewtonBoundedPinkInverseKinematicsAction,
                    )

                    assert cfg.actions.upper_body_ik.class_type is G1NewtonBoundedPinkInverseKinematicsAction
            elif not standalone:
                assert cfg.actions.upper_body_ik.class_type == original_action_type
        raise ConstructionReached

    monkeypatch.setattr(ManagerBasedRLEnv, "__init__", inspect_before_simulation)
    with pytest.raises(ConstructionReached):
        gym.make(task_name, cfg=cfg)
    with pytest.raises(ConstructionReached):
        instantiate(cfg)


@pytest.mark.parametrize(
    "task_name",
    [
        "IsaacContrib-PickPlace-GR1T2-Abs",
        "IsaacContrib-PickPlace-GR1T2-WaistEnabled-Abs",
        "IsaacContrib-PickPlace-Locomanipulation-G1-Abs",
        "IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs",
    ],
)
@pytest.mark.parametrize("physics", ["ovphysx", "newton_mjwarp"])
def test_humanoid_preparation_preserves_caller_changes_and_rejects_backend_switch(monkeypatch, task_name, physics):
    """Teleop preparation, copies and registered construction share one completed preparation."""
    import gymnasium as gym
    from isaaclab_physx.physics import PhysxCfg

    from isaaclab.envs import ManagerBasedRLEnv

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(task_name, device="cpu", num_envs=1, overrides=[f"physics={physics}"])
    controlled_joints = list(cfg.actions.upper_body_ik.pink_controlled_joint_names)
    getattr(cfg, f"stabilize_for_kitless_{physics}")()
    cfg.isaac_teleop.require_initial_bimanual_hand_tracking = (
        not cfg.isaac_teleop.require_initial_bimanual_hand_tracking
    )
    policy = {
        name: getattr(cfg.isaac_teleop, name)
        for name in ("require_initial_bimanual_hand_tracking", "action_conditioner", "action_validator")
    }
    haptics = cfg.haptic_feedback.to_dict()
    images = getattr(cfg, "image_obs_list", None)
    if images is not None:
        images.append("caller_camera")
        images = list(images)
    cfg.sim.dt = 0.003
    actuator_name = next(iter(cfg.scene.robot.actuators))
    cfg.scene.robot.actuators[actuator_name].stiffness = 123.0
    if physics == "newton_mjwarp":
        cfg.sim.physics.solver_cfg.nconmax = 5123
    if "GR1T2" in task_name:
        cfg.actions.upper_body_ik.controller.urdf_path = "caller-selected.urdf"

    class ConstructionReached(Exception):
        pass

    def inspect_before_simulation(self, cfg, *args, **kwargs):
        self._is_closed = True
        assert cfg.actions.upper_body_ik.pink_controlled_joint_names == controlled_joints
        assert all(getattr(cfg.isaac_teleop, name) == value for name, value in policy.items())
        assert cfg.haptic_feedback.to_dict() == haptics
        assert getattr(cfg, "image_obs_list", None) == images
        assert cfg.sim.dt == 0.003
        assert cfg.scene.robot.actuators[actuator_name].stiffness == 123.0
        if physics == "newton_mjwarp":
            assert cfg.sim.physics.solver_cfg.nconmax == 5123
        if "GR1T2" in task_name:
            assert cfg.actions.upper_body_ik.controller.urdf_path == "caller-selected.urdf"
        raise ConstructionReached

    monkeypatch.setattr(ManagerBasedRLEnv, "__init__", inspect_before_simulation)
    for prepared in (cfg, cfg.copy()):
        getattr(prepared, f"stabilize_for_kitless_{physics}")()
        serialized = prepared.to_dict()
        assert serialized["haptic_feedback"] == haptics
        if "GR1T2" in task_name:
            assert serialized["scene"]["robot"]["spawn"]["func"].endswith(":_spawn_and_condition")
        with pytest.raises(ConstructionReached):
            gym.make(task_name, cfg=prepared)
        prepared.sim.physics = PhysxCfg()
        with pytest.raises(ValueError, match="resolve a fresh task configuration"):
            gym.make(task_name, cfg=prepared)


def _spawn_task_roundtrip_fixture(path, _cfg, **_kwargs):
    """Module-level source spawner, preserving the public callable serialization contract."""
    from pxr import UsdGeom, UsdPhysics

    from isaaclab.sim import get_current_stage

    stage = get_current_stage()
    root = stage.DefinePrim(path, "Xform")
    joint = UsdPhysics.RevoluteJoint.Define(stage, path + "/R_thumb_distal_joint")
    joint.CreateLowerLimitAttr().Set(-15.0)
    joint.CreateUpperLimitAttr().Set(80.0)
    table = stage.DefinePrim(path + "/Table", "Xform")
    UsdPhysics.CollisionAPI.Apply(table)
    UsdPhysics.MeshCollisionAPI.Apply(table).CreateApproximationAttr().Set("boundingCube")
    UsdGeom.Mesh.Define(stage, path + "/Table/Mesh")
    for name in ("sm_steeringwheel_a01_wheel_rim_01", "hub_detail"):
        mesh = UsdGeom.Mesh.Define(stage, path + "/" + name).GetPrim()
        UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr().Set("convexDecomposition")
    return root


@pytest.mark.parametrize(
    "task_name",
    [
        "IsaacContrib-PickPlace-GR1T2-Abs",
        "IsaacContrib-PickPlace-GR1T2-WaistEnabled-Abs",
        "IsaacContrib-PickPlace-Locomanipulation-G1-Abs",
        "IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs",
    ],
)
@pytest.mark.parametrize("physics", ["ovphysx", "newton_mjwarp"])
@pytest.mark.parametrize("restore_target", ["copy", "fresh_prepared"])
def test_prepared_task_spawn_callbacks_survive_public_config_roundtrip(monkeypatch, task_name, physics, restore_target):
    """Restored prepared configurations reach registered construction with resolvable spawners."""
    import gymnasium as gym

    from pxr import Usd, UsdPhysics

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Asset, AssetBaseCfg
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils import to_dict, update_from_dict
    from isaaclab.utils.string import string_to_callable

    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(task_name, device="cpu", num_envs=1, overrides=[f"physics={physics}"])
    for asset in (cfg.scene.robot, cfg.scene.packing_table, cfg.scene.object):
        asset.spawn.func = _spawn_task_roundtrip_fixture
    getattr(cfg, f"stabilize_for_kitless_{physics}")()
    cfg.sim.dt = 0.003
    serialized = to_dict(cfg)
    if "GR1T2" in task_name:
        cfg.actions.upper_body_ik.hand_target_margin_rad = 0.123
    if restore_target == "copy":
        restored = cfg.copy()
    else:
        restored = parse_env_cfg(task_name, device="cpu", num_envs=1, overrides=[f"physics={physics}"])
        getattr(restored, f"stabilize_for_kitless_{physics}")()
    update_from_dict(restored, serialized)
    if "GR1T2" in task_name:
        assert cfg.actions.upper_body_ik.hand_target_margin_rad == 0.123

    class ConstructionReached(Exception):
        pass

    def inspect_before_simulation(self, cfg, *args, **kwargs):
        self._is_closed = True
        assert cfg.sim.dt == 0.003
        assets = [cfg.scene.packing_table]
        if physics == "newton_mjwarp":
            assets.append(cfg.scene.object)
        if "GR1T2" in task_name:
            assets.append(cfg.scene.robot)
        stage = Usd.Stage.CreateInMemory()
        for index, asset in enumerate(assets):
            callback = to_dict(asset.spawn)["func"]
            assert callable(string_to_callable(callback))
            path = f"/World/Asset_{index}"
            with sim_utils.use_stage(stage):
                authored = Asset(AssetBaseCfg(prim_path=path, spawn=asset.spawn))
            root = authored.prim
            assert root == stage.GetPrimAtPath(path)
            if asset is cfg.scene.packing_table:
                assert not stage.GetPrimAtPath(path + "/Table").HasAPI(UsdPhysics.CollisionAPI)
                assert stage.GetPrimAtPath(path + "/Table/Mesh").HasAPI(UsdPhysics.CollisionAPI)
            elif asset is cfg.scene.object:
                assert (
                    stage.GetPrimAtPath(path + "/hub_detail").GetAttribute("physics:approximation").Get()
                    == "convexHull"
                )
                assert (
                    stage.GetPrimAtPath(path + "/sm_steeringwheel_a01_wheel_rim_01")
                    .GetAttribute("physics:approximation")
                    .Get()
                    == "convexDecomposition"
                )
            else:
                limits = authored.cfg.spawn.action_cfg.hand_joint_limits["R_thumb_distal_joint"]
                assert limits == pytest.approx((-0.2617993878, 1.3962634016))
        if "GR1T2" in task_name:
            assert cfg.scene.robot.spawn.action_cfg is cfg.actions.upper_body_ik
        raise ConstructionReached

    monkeypatch.setattr(sim_utils, "find_matching_prim_paths", lambda path: [path])
    monkeypatch.setattr(ManagerBasedRLEnv, "__init__", inspect_before_simulation)
    with pytest.raises(ConstructionReached):
        gym.make(task_name, cfg=restored)
