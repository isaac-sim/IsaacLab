# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Architecture gates for task-free golden rendering tests."""

import ast
import sys
from pathlib import Path

from isaaclab.renderers.output_contract import RenderBufferKind
from isaaclab.test.integration_scene_cfgs import RenderingTestSceneCfg

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RENDERER_TEST_DIR = Path(__file__).parent / "renderers"
_VISUALIZER_TEST_DIR = _REPO_ROOT / "source" / "isaaclab_visualizers" / "test"
_TASK_TEST_DIR = _REPO_ROOT / "source" / "isaaclab_tasks" / "test"
_CORE_PACKAGE = _REPO_ROOT / "source" / "isaaclab" / "isaaclab"
sys.path.insert(0, str(_RENDERER_TEST_DIR))

from rendering_cases import (  # noqa: E402
    KIT_CASES,
    KIT_RENDERING_CASES,
    KITLESS_CASES,
    KITLESS_RENDERING_CASES,
    OVRTX_AOVS,
    SCENE_PROBE_KIT_CASES,
    SCENE_PROBE_KITLESS_CASES,
    SIMPLE_SHADING_AOVS,
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
    """Keep exactly one Kit and one Kit-less collection root and reject retired ownership."""
    forbidden_paths = [
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

    entrypoints = sorted(_RENDERER_TEST_DIR.glob("test_rendering*.py"))
    assert {path.name for path in entrypoints} == {"test_rendering_kit.py", "test_rendering_kitless.py"}
    for path in entrypoints:
        source = path.read_text()
        assert "def test_" not in source
        assert "pytest.mark.cold_cache" in source
        factory = "make_kitless_test()" if path.name.endswith("kitless.py") else "make_kit_test()"
        assert source.count(factory) == 1

    assert {path.name for path in (_VISUALIZER_TEST_DIR / "golden_images").iterdir()} == {"rendering_scene"}
    active_config = "\n".join(
        path.read_text()
        for path in (
            _REPO_ROOT / ".github" / "workflows" / "build.yaml",
            _REPO_ROOT / ".github" / "test-subsets" / "postmerge-rendering.toml",
            _REPO_ROOT / "tools" / "test_settings.py",
        )
    )
    stale_names = (
        "rendering_test_utils.py",
        "test_rendering_cartpole.py",
        "test_rendering_scene_probes_",
        "test_rendering_kitless_legacy_",
        "test_rendering_kitless_ovstage_",
        "test_visualizer_golden_newton.py",
        "visualizer_integration_utils.py",
    )
    assert not [name for name in stale_names if name in active_config]
    assert 'filter-pattern: "not isaaclab_"\n        exclude-pattern: "test_rendering_"' not in active_config
    assert "franka_cloth-ovphysx" not in active_config


def test_rendering_native_process_boundaries_are_complete() -> None:
    """Every case is selected by exactly one bounded fresh-process partition."""
    settings_tree = ast.parse((_REPO_ROOT / "tools" / "test_settings.py").read_text())
    assigned_names = {
        target.id
        for node in settings_tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assignments = {
        target.id: ast.literal_eval(node.value)
        for node in settings_tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id == "PROCESS_ISOLATED_TESTS"
    }
    assert "COLD_CACHE_TESTS" not in assigned_names

    case_ids_by_entrypoint = {
        "source/isaaclab/test/renderers/test_rendering_kit.py": [case.id for case in KIT_RENDERING_CASES],
        "source/isaaclab/test/renderers/test_rendering_kitless.py": [
            f"{stage}-{case.id}" for stage, case in KITLESS_RENDERING_CASES
        ],
    }
    process_partitions = assignments["PROCESS_ISOLATED_TESTS"]
    assert set(process_partitions) == set(case_ids_by_entrypoint)
    for entrypoint, case_ids in case_ids_by_entrypoint.items():
        assert len(case_ids) == len(set(case_ids))
        partitions = process_partitions[entrypoint]
        selected_counts = {name: 0 for name, _ in partitions}
        for case_id in case_ids:
            matches = [name for name, selectors in partitions if any(selector in case_id for selector in selectors)]
            assert len(matches) == 1, f"{case_id} is selected by {matches}"
            selected_counts[matches[0]] += 1
        assert all(selected_counts.values())
        assert max(selected_counts.values()) <= 15, (
            "Each partition builds one scene per case and has a 15-scene budget."
        )

    kitless_runner_source = (_RENDERER_TEST_DIR / "kitless_rendering_runner.py").read_text()
    thread_limit = 'os.environ["PXR_WORK_THREAD_LIMIT"] = "1"'
    for rendering_import in ("from rendering_cases import", "from rendering_runner import"):
        assert kitless_runner_source.index(thread_limit) < kitless_runner_source.index(rendering_import)
    assert thread_limit not in (_RENDERER_TEST_DIR / "rendering_runner.py").read_text()
    assert 'monkeypatch.setenv("PXR_WORK_THREAD_LIMIT"' not in kitless_runner_source


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
    """Specialized geometry owns scene facts while the shared runner remains generic."""
    path = _RENDERER_TEST_DIR / "rendering_scene_cfgs.py"
    tree = ast.parse(path.read_text())
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert {name for name in classes if name.endswith("RenderingSceneCfg")} == {
        "FrankaClothRenderingSceneCfg",
        "FrankaSoftRenderingSceneCfg",
        "KukaHeterogeneousRenderingSceneCfg",
        "ShadowHandRenderingSceneCfg",
    }
    assert "RenderingSceneSpec" in classes
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
    assert KIT_RENDERING_CASES == KIT_CASES + SCENE_PROBE_KIT_CASES
    assert KITLESS_RENDERING_CASES == KITLESS_CASES + SCENE_PROBE_KITLESS_CASES

    runner_source = (_RENDERER_TEST_DIR / "rendering_runner.py").read_text()
    assert not [scene for scene in expected_scenes if f'"{scene}"' in runner_source]
    assert "make_rendering_scene_cfg" not in runner_source

    rendering_utils = ast.parse((_CORE_PACKAGE / "test" / "utils" / "rendering.py").read_text())
    functions = {node.name for node in rendering_utils.body if isinstance(node, ast.FunctionDef)}
    assert "make_rendering_physics_cfg" in functions
    assert "make_physics_cfg" not in functions


def test_legacy_shadow_semantics_stay_in_canonical_scene() -> None:
    """Do not golden-test the GPU-dependent source/clone split in the specialized legacy scene."""
    legacy_shadow_warp = [
        case
        for stage, case in SCENE_PROBE_KITLESS_CASES
        if stage == "legacy" and case.scene == "shadow_hand" and case.renderer == "newton_warp"
    ]
    assert {case.physics for case in legacy_shadow_warp} == {"newton", "ovphysx"}
    assert all(
        case.aovs == (RenderBufferKind.RGB, RenderBufferKind.INSTANCE_SEGMENTATION) for case in legacy_shadow_warp
    )
    kit_shadow_warp = [
        case for case in SCENE_PROBE_KIT_CASES if case.scene == "shadow_hand" and case.renderer == "newton_warp"
    ]
    assert {case.physics for case in kit_shadow_warp} == {"newton", "physx"}
    assert all(RenderBufferKind.SEMANTIC_SEGMENTATION in case.aovs for case in kit_shadow_warp)
    canonical_legacy_warp = [
        case for stage, case in KITLESS_CASES if stage == "legacy" and case.renderer == "newton_warp"
    ]
    assert {case.physics for case in canonical_legacy_warp} == {"newton", "ovphysx"}
    assert all(RenderBufferKind.SEMANTIC_SEGMENTATION in case.aovs for case in canonical_legacy_warp)


def test_renderer_matrix_bundles_compatible_aovs() -> None:
    """The matrix bundles compatible AOVs, uses one vocabulary, and stays within its cost budget."""
    assert len(KIT_CASES) <= 10, "Each Kit case builds one scene and commits one baseline set."
    assert len(KITLESS_CASES) <= 50, "OVRTX AOV isolation makes each Kit-less case a separate scene build."
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

    ovstage_cases = [case for stage, case in KITLESS_CASES if stage == "ovstage"]
    assert {case.physics for case in ovstage_cases} == {"ovphysx", "newton"}
    assert all(case.renderer == "ovrtx" for case in ovstage_cases)

    all_cases = [*KIT_RENDERING_CASES, *(case for _, case in KITLESS_RENDERING_CASES)]
    assert all(isinstance(aov, RenderBufferKind) for case in all_cases for aov in case.aovs)
    simple_profiles = {aov.value for aov in SIMPLE_SHADING_AOVS}
    assert {case.profile for case in KIT_CASES if case.profile in simple_profiles} == simple_profiles
    assert {case.profile for _, case in KITLESS_CASES if case.profile in simple_profiles} == simple_profiles
    for case in all_cases:
        assert len(set(case.aovs) & set(SIMPLE_SHADING_AOVS)) <= 1
        if case.profile in simple_profiles:
            assert case.aovs == (RenderBufferKind(case.profile),)


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
    renderer_expected = {case.scene: set() for case in KIT_RENDERING_CASES}
    renderer_expected.update({case.scene: set() for _, case in KITLESS_RENDERING_CASES})
    for case in KIT_RENDERING_CASES:
        renderer_expected[case.scene].update(
            f"kit-{case.golden_id(aov)}.png" for aov in case.aovs if aov != RenderBufferKind.MOTION_VECTORS
        )
    for stage, case in KITLESS_RENDERING_CASES:
        renderer_expected[case.scene].update(
            f"{stage}-{case.golden_id(aov)}.png" for aov in case.aovs if aov != RenderBufferKind.MOTION_VECTORS
        )

    renderer_root = _RENDERER_TEST_DIR / "golden_images"
    assert {path.name for path in renderer_root.iterdir() if path.is_dir()} == set(renderer_expected)
    for scene, expected in renderer_expected.items():
        assert {path.name for path in (renderer_root / scene).glob("*.png")} == expected
    assert not list(renderer_root.rglob("*motion_vectors.png"))

    visualizer_expected = {
        f"{physics}-{visualizer}-{mode}.png"
        for physics in ("physx", "newton")
        for visualizer in ("kit", "newton")
        for mode in ("viewport", "tiled")
    }
    visualizer_dir = _VISUALIZER_TEST_DIR / "golden_images" / "rendering_scene"
    assert {path.name for path in visualizer_dir.glob("*.png")} == visualizer_expected
