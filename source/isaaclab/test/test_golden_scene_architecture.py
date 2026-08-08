# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Architecture gates for task-free golden rendering tests."""

import ast
import sys
from pathlib import Path

from isaaclab.test.integration_scene_cfgs import RenderingTestSceneCfg

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RENDERER_TEST_DIR = Path(__file__).parent / "renderers"
_VISUALIZER_TEST_DIR = _REPO_ROOT / "source" / "isaaclab_visualizers" / "test"
_TASK_TEST_DIR = _REPO_ROOT / "source" / "isaaclab_tasks" / "test"
_CORE_PACKAGE = _REPO_ROOT / "source" / "isaaclab" / "isaaclab"
sys.path.insert(0, str(_RENDERER_TEST_DIR))

from rendering_cases import (  # noqa: E402
    KIT_CASES,
    KITLESS_CASES,
    OVRTX_AOVS,
    SCENE_PROBE_KIT_CASES,
    SCENE_PROBE_KITLESS_CASES,
    SIMPLE_SHADING_AOVS,
    select_kitless_cases,
    select_kitless_scene_probe_cases,
)


def test_golden_ownership_has_no_task_or_environment_dependencies() -> None:
    """Golden infrastructure may depend on scenes and simulation, never RL environments."""
    files = [
        *sorted(_RENDERER_TEST_DIR.glob("*.py")),
        _VISUALIZER_TEST_DIR / "test_visualizer_rendering.py",
        _VISUALIZER_TEST_DIR / "visualizer_test_utils.py",
        _CORE_PACKAGE / "test" / "integration_scene_cfgs.py",
        _CORE_PACKAGE / "test" / "utils" / "golden_image.py",
        _CORE_PACKAGE / "test" / "utils" / "rendering.py",
    ]
    forbidden = ("isaaclab_tasks", "gymnasium", "ManagerBasedEnv", "DirectRLEnv", "hydra")
    violations = {
        str(path.relative_to(_REPO_ROOT)): [token for token in forbidden if token in path.read_text()]
        for path in files
        if any(token in path.read_text() for token in forbidden)
    }
    assert not violations


def test_removed_task_and_visualizer_harnesses_stay_removed() -> None:
    """Reject the former task-owned helpers, scenes, goldens, and retry configuration."""
    forbidden_paths = [
        _RENDERER_TEST_DIR / "test_rendering_kitless.py",
        _TASK_TEST_DIR / "rendering_test_utils.py",
        _TASK_TEST_DIR / "test_maybe_save_stage_golden.py",
        _TASK_TEST_DIR / "test_parametrization_helpers.py",
        _TASK_TEST_DIR / "golden_images",
        _TASK_TEST_DIR / "golden_stages",
        _VISUALIZER_TEST_DIR / "visualizer_golden_utils.py",
        _VISUALIZER_TEST_DIR / "visualizer_integration_utils.py",
        *(_TASK_TEST_DIR / "core").glob("test_rendering*.py"),
        *_VISUALIZER_TEST_DIR.glob("test_visualizer_*_newton.py"),
        *_VISUALIZER_TEST_DIR.glob("test_visualizer_*_physx.py"),
    ]
    assert not [path for path in forbidden_paths if path.exists()]
    partition_paths = sorted(_RENDERER_TEST_DIR.glob("test_rendering_kitless_*.py"))
    assert {path.name for path in partition_paths} == {
        "test_rendering_kitless_legacy_newton.py",
        "test_rendering_kitless_legacy_ovphysx.py",
        "test_rendering_kitless_ovstage_newton.py",
        "test_rendering_kitless_ovstage_ovphysx.py",
    }
    assert all(path.read_text().count("make_kitless_test(") == 1 for path in partition_paths)
    assert not any("def test_rendering_scene_kitless" in path.read_text() for path in partition_paths)
    probe_partition_paths = sorted(_RENDERER_TEST_DIR.glob("test_rendering_scene_probes_kitless_*.py"))
    assert {path.name for path in probe_partition_paths} == {
        "test_rendering_scene_probes_kitless_legacy_newton.py",
        "test_rendering_scene_probes_kitless_legacy_ovphysx.py",
        "test_rendering_scene_probes_kitless_ovstage_ovphysx.py",
    }
    assert all(path.read_text().count("make_kitless_test(") == 1 for path in probe_partition_paths)
    assert all("scene_probes=True" in path.read_text() for path in probe_partition_paths)
    assert not any("def test_" in path.read_text() for path in [*partition_paths, *probe_partition_paths])
    assert "def test_" not in (_RENDERER_TEST_DIR / "test_rendering_scene_probes_kit.py").read_text()

    assert {path.name for path in (_VISUALIZER_TEST_DIR / "golden_images").iterdir()} == {"rendering_scene"}

    active_config = "\n".join(
        path.read_text()
        for path in (
            _REPO_ROOT / ".github" / "workflows" / "build.yaml",
            _REPO_ROOT / ".github" / "test-subsets" / "postmerge-rendering.toml",
            _REPO_ROOT / "tools" / "conftest.py",
            _REPO_ROOT / "tools" / "test_settings.py",
        )
    )
    stale_names = (
        "rendering_test_utils.py",
        "test_rendering_cartpole.py",
        "test_rendering_kitless.py",
        "test_visualizer_golden_newton.py",
        "visualizer_integration_utils.py",
    )
    assert not [name for name in stale_names if name in active_config]
    assert 'filter-pattern: "not isaaclab_"\n        exclude-pattern: "test_rendering_"' not in active_config


def test_one_scene_configuration_owns_deliberate_composition() -> None:
    """The canonical scene has one owner, purposeful placement, defaults, and labels."""
    definitions = [
        path.relative_to(_REPO_ROOT)
        for path in (_REPO_ROOT / "source").rglob("*.py")
        if any(
            isinstance(node, ast.ClassDef) and node.name == "RenderingTestSceneCfg"
            for node in ast.parse(path.read_text()).body
        )
    ]
    assert definitions == [Path("source/isaaclab/isaaclab/test/integration_scene_cfgs.py")]

    cfg = RenderingTestSceneCfg(num_envs=1, env_spacing=5.0)
    composition_positions = {
        tuple(cfg.robot.init_state.pos),
        tuple(cfg.moving_cube.init_state.pos),
        tuple(cfg.table.init_state.pos),
    }
    assert len(composition_positions) == 3
    assert (0.0, 0.0, 0.0) not in composition_positions
    assert cfg.cylinder.init_state.pos[2] > cfg.table.init_state.pos[2]
    assert cfg.sphere.init_state.pos[2] > cfg.table.init_state.pos[2]
    assert cfg.robot.init_state.joint_pos == {"slider_to_cart": -0.25, "cart_to_pole": 0.45}
    assert type(cfg.ground.spawn).__name__ == "CuboidCfg"

    for name in ("ground", "robot", "moving_cube", "table", "cylinder", "sphere"):
        assert ("class", name) in getattr(cfg, name).spawn.semantic_tags


def test_specialized_rendering_scenes_are_declarative_and_task_free() -> None:
    """Specialized geometry lives in scene configs while shared code owns test behavior."""
    path = _RENDERER_TEST_DIR / "rendering_scene_cfgs.py"
    tree = ast.parse(path.read_text())
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert classes == {
        "FrankaClothRenderingSceneCfg",
        "FrankaSoftRenderingSceneCfg",
        "KukaHeterogeneousRenderingSceneCfg",
        "ShadowHandRenderingSceneCfg",
    }
    assert not [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    ]
    source = path.read_text()
    assert not [token for token in ("isaaclab_tasks", "gymnasium", "isaaclab.envs", "hydra") if token in source]

    expected_scenes = {"franka_cloth", "franka_soft", "kuka_heterogeneous", "shadow_hand"}
    assert {case.scene for case in SCENE_PROBE_KIT_CASES} == expected_scenes
    assert {case.scene for _, case in SCENE_PROBE_KITLESS_CASES} == expected_scenes - {"kuka_heterogeneous"}
    partitions = [
        select_kitless_scene_probe_cases(stage, physics)
        for stage in ("legacy", "ovstage")
        for physics in ("ovphysx", "newton")
    ]
    assert tuple(map(len, partitions)) == (4, 10, 3, 0)
    assert sum(map(len, partitions)) == len(SCENE_PROBE_KITLESS_CASES)
    assert set().union(*partitions) == set(SCENE_PROBE_KITLESS_CASES)


def test_specialized_rendering_scenes_preserve_task_signatures() -> None:
    """Direct probes retain recognizable task assets and layouts without task ownership."""
    source = (_RENDERER_TEST_DIR / "rendering_scene_cfgs.py").read_text()
    classes = {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name.endswith("RenderingSceneCfg")
    }
    fields = {}
    for name, node in classes.items():
        fields[name] = set()
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign):
                targets = (statement.target,)
            elif isinstance(statement, ast.Assign):
                targets = statement.targets
            else:
                continue
            fields[name].update(target.id for target in targets if isinstance(target, ast.Name))

    assert fields == {
        "FrankaSoftRenderingSceneCfg": {
            "ground",
            "key_light",
            "fill_light",
            "camera",
            "robot",
            "table",
            "deformable",
        },
        "FrankaClothRenderingSceneCfg": {
            "ground",
            "key_light",
            "fill_light",
            "camera",
            "robot",
            "table",
            "support_neg_y",
            "support_pos_y",
            "deformable",
        },
        "KukaHeterogeneousRenderingSceneCfg": {
            "ground",
            "key_light",
            "fill_light",
            "camera",
            "robot",
            "object",
            "table",
        },
        "ShadowHandRenderingSceneCfg": {"ground", "key_light", "fill_light", "camera", "robot", "object"},
    }

    required_signatures = (
        "FRANKA_PANDA_MENAGERIE_CFG",
        "DexCube/dex_cube_instanceable.usd",
        "size=(1.3, 0.9, 1.05)",
        "pos=(0.5, 0.0, -0.525)",
        "diffuse_color=(0.8, 0.5, 0.5)",
        "size=(0.3, 0.04, 0.04)",
        "edge_refinement=3.0",
        "diffuse_color=(0.45, 0.45, 0.85)",
        "size=(0.2, 0.2)",
        "resolution=(8, 8)",
        "pos=(0.4, 0.0, 0.102)",
        "rot=(0.70710678, 0.0, 0.0, 0.70710678)",
        "size=(0.1, 0.02, 0.15)",
        "pos=(0.4, -0.02, 0.075)",
        "pos=(0.4, 0.02, 0.075)",
        '"panda_finger2_passive": ImplicitActuatorCfg(',
        "stiffness=350.0",
        "NewtonCollisionPipelineCfg(enable_rigid_soft_full_surface_contact=True)",
        "soft_contact_ke=8.0e3, soft_contact_mu=10.0",
        "pos=(-0.55, 0.1, 0.35)",
        'joint_pos={".*": 0.0}',
        "rot=(0.5080, 0.2114, 0.318, 0.7720)",
        "clipping_range=(0.01, 3.0)",
        "rot=(0.6124, 0.3536, 0.3536, 0.6124)",
        "clipping_range=(0.01, 2.5)",
        "width=64",
        "random_choice=False",
        "diffuse_color=(0.25, 0.15, 0.15)",
        "rot=(0.0, 0.7071, 0.0, 0.7071)",
        'frozenset({"cube"}), None, frozenset({"robot"})',
        "width=120",
        "height=120",
        "clipping_range=(0.1, 20.0)",
        "num_envs=4, env_spacing=2.0",
        "num_envs=4, env_spacing=3.0",
        "_SOFT_NEWTON_PHYSICS, 2.0",
        "_CLOTH_NEWTON_PHYSICS",
        "2.5",
        "(0.0, -0.35, 1.0), (0.0, -0.35, 0.0)",
        "retrieve_file_path(cfg.fill_light.spawn.texture_file)",
        "MeshCuboidCfg",
        "MeshSphereCfg",
        "MeshCapsuleCfg",
        "MeshConeCfg",
    )
    assert not [signature for signature in required_signatures if signature not in source]

    rejected_demo_composition = (
        "from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG\n",
        "PhysxDeformableBodyPropertiesCfg",
        "_SOFT_PHYSX",
        "_CLOTH_PHYSX",
        "SeattleLabTable/table_instanceable.usd",
        "_FRANKA_CLOTH_CUBE",
        "size=(0.3, 0.05, 0.05)",
        "resolution=(12, 12)",
        "size=(0.03, 0.01, 0.08)",
        "_YOUNGS_MODULUS = 8.0e4",
        "hand_actuator =",
        "pos=(0.4, 0.0, 0.2)",
        "_SHADOW_HAND_JOINT_POS",
        'prim_path="{ENV_REGEX_NS}/Support"',
        "pos=(-0.7, -0.25, 0.0)",
        "pos=(-0.65, -0.2, 0.0)",
        "size=(0.32, 0.18, 0.16)",
        "size=(0.55, 0.55)",
        "resolution=(30, 30)",
        "SHADOW_HAND_NEWTON_CFG.init_state.replace",
    )
    assert not [signature for signature in rejected_demo_composition if signature in source]


def test_renderer_matrix_bundles_compatible_aovs() -> None:
    """The matrix bundles compatible AOVs and isolates mutually exclusive profiles."""
    assert len(KIT_CASES) == 8
    assert len(KITLESS_CASES) == 46
    standard_cases = [case for case in KIT_CASES if case.profile == "standard"]
    standard_cases.extend(case for _, case in KITLESS_CASES if case.profile == "standard")
    assert all(len(case.aovs) == 1 for case in standard_cases if case.renderer == "ovrtx")
    assert all(len(case.aovs) > 1 for case in standard_cases if case.renderer != "ovrtx")
    for physics in ("ovphysx", "newton"):
        actual = tuple(
            case.aovs[0]
            for stage, case in KITLESS_CASES
            if stage == "legacy" and case.physics == physics and case.renderer == "ovrtx" and case.profile == "standard"
        )
        assert actual == OVRTX_AOVS

    partitions = [
        select_kitless_cases(stage, physics) for stage in ("legacy", "ovstage") for physics in ("ovphysx", "newton")
    ]
    assert max(map(len, partitions)) == 14
    assert sum(map(len, partitions)) == len(KITLESS_CASES)
    assert set().union(*partitions) == set(KITLESS_CASES)

    all_cases = [*KIT_CASES, *(case for _, case in KITLESS_CASES)]
    simple_aovs = set(SIMPLE_SHADING_AOVS)
    assert {case.profile for case in KIT_CASES if case.profile in simple_aovs} == simple_aovs
    assert {case.profile for _, case in KITLESS_CASES if case.profile in simple_aovs} == simple_aovs
    for case in all_cases:
        assert len(set(case.aovs) & simple_aovs) <= 1
        if case.profile in simple_aovs:
            assert case.aovs == (case.profile,)


def test_rendering_scene_closes_scene_before_simulation_teardown() -> None:
    """The direct composition root isolates RTX history and closes entities before the sim."""
    module = ast.parse((_CORE_PACKAGE / "test" / "utils" / "rendering.py").read_text())
    rendering_scene = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "RenderingScene"
    )
    stabilize = next(
        node for node in rendering_scene.body if isinstance(node, ast.FunctionDef) and node.name == "stabilize_camera"
    )
    calls = {ast.unparse(node) for node in ast.walk(stabilize) if isinstance(node, ast.Call)}
    assert "omni.usd.get_context().reset_renderer_accumulation()" in calls

    build = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "build_rendering_scene"
    )
    cleanup = next(node for node in ast.walk(build) if isinstance(node, ast.Try))
    assert [ast.unparse(node) for node in cleanup.finalbody] == ["runtime.scene.close()"]


def test_rendering_reset_preserves_scene_owned_fixed_roots() -> None:
    """The shared runner delegates configured-state restoration to the MDP event."""
    module = ast.parse((_CORE_PACKAGE / "test" / "utils" / "rendering.py").read_text())
    rendering_scene = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "RenderingScene"
    )
    reset = next(node for node in rendering_scene.body if isinstance(node, ast.FunctionDef) and node.name == "reset")
    call = next(
        node
        for node in ast.walk(reset)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "reset_scene_to_default"
    )
    assert [ast.unparse(arg) for arg in call.args] == ["SimpleNamespace(scene=self.scene)", "env_ids"]
    assert {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords} == {
        "reset_joint_targets": "True",
        "preserve_fixed_articulation_roots": "self.preserve_fixed_articulation_roots",
    }


def test_reset_policy_remains_in_mdp() -> None:
    """InteractiveScene must not own the MDP policy for restoring configured state."""
    scene_source = (_CORE_PACKAGE / "scene" / "interactive_scene.py").read_text()
    assert "def reset_to_default(" not in scene_source

    events_path = _CORE_PACKAGE / "envs" / "mdp" / "events.py"
    function = next(
        node
        for node in ast.parse(events_path.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "reset_scene_to_default"
    )
    loops = [node for node in function.body if isinstance(node, ast.For)]
    assert [ast.unparse(loop.iter) for loop in loops] == [
        "env.scene.rigid_objects.values()",
        "env.scene.articulations.items()",
        "env.scene.cable_objects.values()",
        "env.scene.deformable_objects.values()",
    ]
    calls = {ast.unparse(node.func) for node in ast.walk(function) if isinstance(node, ast.Call)}
    assert {
        "rigid_object.write_root_pose_to_sim_index",
        "articulation_asset.write_joint_position_to_sim_index",
        "cable_object.write_segment_pose_to_sim_index",
        "deformable_object.write_nodal_state_to_sim",
    } <= calls


def test_golden_inventory_matches_case_matrix() -> None:
    """Checked-in baselines exactly match the declared renderer and visualizer matrices."""
    renderer_expected = {
        scene: set()
        for scene in ("rendering_scene", "franka_cloth", "franka_soft", "kuka_heterogeneous", "shadow_hand")
    }
    for case in (*KIT_CASES, *SCENE_PROBE_KIT_CASES):
        renderer_expected[case.scene].update(f"kit-{case.golden_id(aov)}.png" for aov in case.aovs)
    for stage, case in (*KITLESS_CASES, *SCENE_PROBE_KITLESS_CASES):
        renderer_expected[case.scene].update(f"{stage}-{case.golden_id(aov)}.png" for aov in case.aovs)

    renderer_root = _RENDERER_TEST_DIR / "golden_images"
    assert {path.name for path in renderer_root.iterdir() if path.is_dir()} == set(renderer_expected)
    for scene, expected in renderer_expected.items():
        assert {path.name for path in (renderer_root / scene).glob("*.png")} == expected

    visualizer_expected = {
        f"{physics}-{visualizer}-{mode}.png"
        for physics in ("physx", "newton")
        for visualizer in ("kit", "newton")
        for mode in ("viewport", "tiled")
    }
    visualizer_dir = _VISUALIZER_TEST_DIR / "golden_images" / "rendering_scene"
    assert {path.name for path in visualizer_dir.glob("*.png")} == visualizer_expected
