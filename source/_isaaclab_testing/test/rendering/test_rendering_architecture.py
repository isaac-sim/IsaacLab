# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Architecture gates for rendering in the private downstream testing project."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import tomllib
from packaging.requirements import Requirement

from isaaclab.renderers.output_contract import RenderBufferKind

_SUITE_DIR = Path(__file__).resolve().parent
_TEST_ROOT = _SUITE_DIR.parent
_PROJECT_ROOT = _TEST_ROOT.parent
_REPO_ROOT = _PROJECT_ROOT.parents[1]
_CORE_ROOT = _REPO_ROOT / "source" / "isaaclab"
_TASK_ROOT = _REPO_ROOT / "source" / "isaaclab_tasks"
_VISUALIZER_ROOT = _REPO_ROOT / "source" / "isaaclab_visualizers"
sys.path.insert(0, str(_SUITE_DIR))

from rendering_cases import (  # noqa: E402
    KIT_CASES,
    KIT_RENDERING_CASES,
    KITLESS_CASES,
    KITLESS_RENDERING_CASES,
    OVRTX_AOVS,
    SIMPLE_SHADING_AOVS,
    SPECIALIZED_KIT_CASES,
    SPECIALIZED_KITLESS_CASES,
    RenderCase,
)


def _assignment(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not assigned in {path}.")


def _defined_classes(path: Path) -> set[str]:
    return {node.name for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef)}


def test_testing_project_owns_rendering_suite() -> None:
    """The private project is a dependency sink whose rendering code stays in one suite."""
    assert _PROJECT_ROOT.name == "_isaaclab_testing"
    assert _SUITE_DIR == _TEST_ROOT / "rendering"
    assert not list(_TEST_ROOT.glob("*.py"))
    former_projects = ("isaaclab_rendering_tests", "_isaaclab_integration_tests")
    assert not [name for name in former_projects if (_REPO_ROOT / "source" / name).exists()]

    with (_PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        manifest = tomllib.load(file)
    dependencies = {Requirement(requirement).name for requirement in manifest["project"]["dependencies"]}
    assert {
        "isaaclab",
        "isaaclab-assets",
        "isaaclab-contrib",
        "isaaclab-newton",
        "isaaclab-ov",
        "isaaclab-ovphysx",
        "isaaclab-physx",
        "isaaclab-visualizers",
    } <= dependencies
    assert manifest["project"]["name"] == "isaaclab-testing"
    assert manifest["tool"]["setuptools"]["packages"] == []
    assert manifest["tool"]["setuptools"]["py-modules"] == []

    old_owners = (
        _CORE_ROOT / "isaaclab/test/utils/golden_image.py",
        _CORE_ROOT / "isaaclab/test/utils/rendering.py",
        _CORE_ROOT / "test/test_golden_scene_architecture.py",
        _CORE_ROOT / "test/renderers/rendering_cases.py",
        _CORE_ROOT / "test/renderers/rendering_runner.py",
        _CORE_ROOT / "test/renderers/rendering_scene_cfgs.py",
        _CORE_ROOT / "test/renderers/test_rendering_kit.py",
        _CORE_ROOT / "test/renderers/test_rendering_kitless.py",
        _CORE_ROOT / "test/renderers/golden_images",
        _VISUALIZER_ROOT / "test/test_visualizer_rendering.py",
        _VISUALIZER_ROOT / "test/visualizer_test_utils.py",
        _VISUALIZER_ROOT / "test/visualizer_golden_utils.py",
        _VISUALIZER_ROOT / "test/visualizer_integration_utils.py",
        _VISUALIZER_ROOT / "test/golden_images/rendering_scene",
        *_VISUALIZER_ROOT.glob("test/test_visualizer_*_newton.py"),
        *_VISUALIZER_ROOT.glob("test/test_visualizer_*_physx.py"),
        _TASK_ROOT / "test/rendering_test_utils.py",
        _TASK_ROOT / "test/test_maybe_save_stage_golden.py",
        _TASK_ROOT / "test/test_parametrization_helpers.py",
        _TASK_ROOT / "test/golden_images",
        _TASK_ROOT / "test/golden_stages",
        *(_TASK_ROOT / "test/core").glob("test_rendering*.py"),
    )
    assert not [path.relative_to(_REPO_ROOT) for path in old_owners if path.exists()]

    project_names = (*former_projects, "_isaaclab_testing", "isaaclab-integration-tests", "isaaclab-testing")
    reverse_references = [
        path.relative_to(_REPO_ROOT)
        for root in (_CORE_ROOT, _TASK_ROOT, _VISUALIZER_ROOT)
        for path in root.rglob("*")
        if path.suffix in {".py", ".toml"} and any(name in path.read_text() for name in project_names)
    ]
    assert not reverse_references


def test_executable_roots_are_compact_and_task_free() -> None:
    """Keep two renderer roots, one visualizer root, and no environment imports."""
    renderer_roots = {
        path.name for path in _SUITE_DIR.glob("test_rendering_*.py") if path.name != "test_rendering_architecture.py"
    }
    assert renderer_roots == {"test_rendering_kit.py", "test_rendering_kitless.py"}
    assert (_SUITE_DIR / "test_visualizer_rendering.py").is_file()
    for name in renderer_roots:
        assert "pytest.mark.cold_cache" in (_SUITE_DIR / name).read_text()

    forbidden = ("isaaclab_tasks", "gymnasium", "ManagerBasedEnv", "DirectRLEnv", "hydra")
    violations = {
        path.name: [token for token in forbidden if token in path.read_text()]
        for path in _SUITE_DIR.glob("*.py")
        if path != Path(__file__)
        if any(token in path.read_text() for token in forbidden)
    }
    assert not violations

    factories = {
        "rendering_runner.py": ("generate_kit_test_cases", "make_kit_test"),
        "kitless_rendering_runner.py": ("generate_kitless_test_cases", "make_kitless_test"),
    }
    for filename, (expected, rejected) in factories.items():
        functions = {
            node.name
            for node in ast.parse((_SUITE_DIR / filename).read_text()).body
            if isinstance(node, ast.FunctionDef)
        }
        assert expected in functions and rejected not in functions


def test_process_partitions_cover_every_rendering_case_once() -> None:
    """Each native scene family runs in exactly one fresh-process partition."""
    settings_path = _REPO_ROOT / "tools/test_settings.py"
    settings_tree = ast.parse(settings_path.read_text())
    assert not any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "COLD_CACHE_TESTS" for target in node.targets)
        for node in settings_tree.body
    )

    cases_by_root = {
        "source/_isaaclab_testing/test/rendering/test_rendering_kit.py": [
            case.id for case in KIT_RENDERING_CASES
        ],
        "source/_isaaclab_testing/test/rendering/test_rendering_kitless.py": [
            f"{stage}-{case.id}" for stage, case in KITLESS_RENDERING_CASES
        ],
    }
    partitions = _assignment(settings_path, "PROCESS_ISOLATED_TESTS")
    assert set(cases_by_root) <= set(partitions)
    for root, case_ids in cases_by_root.items():
        assert len(case_ids) == len(set(case_ids))
        counts = {name: 0 for name, _ in partitions[root]}
        for case_id in case_ids:
            matches = [
                name for name, selectors in partitions[root] if any(selector in case_id for selector in selectors)
            ]
            assert len(matches) == 1, f"{case_id} is selected by {matches}"
            counts[matches[0]] += 1
        assert all(counts.values())
        assert max(counts.values()) <= 15, "Each process has a 15-scene compilation budget."

    kitless_source = (_SUITE_DIR / "kitless_rendering_runner.py").read_text()
    thread_limit = 'os.environ["PXR_WORK_THREAD_LIMIT"] = "1"'
    assert all(
        kitless_source.index(thread_limit) < kitless_source.index(statement)
        for statement in ("from rendering_cases import", "from rendering_runner import")
    )


def test_scene_composition_has_one_downstream_owner() -> None:
    """The shared scene has deliberate placement, defaults, and semantic labels."""
    owners = [
        path.relative_to(_REPO_ROOT)
        for path in (_REPO_ROOT / "source").rglob("*.py")
        if "RenderingTestSceneCfg" in _defined_classes(path)
    ]
    assert owners == [Path("source/_isaaclab_testing/test/rendering/rendering_scene_cfgs.py")]

    from rendering_scene_cfgs import RenderingTestSceneCfg

    cfg = RenderingTestSceneCfg(num_envs=1, env_spacing=5.0)
    positions = {
        tuple(cfg.robot.init_state.pos),
        tuple(cfg.moving_cube.init_state.pos),
        tuple(cfg.table.init_state.pos),
    }
    assert len(positions) == 3 and (0.0, 0.0, 0.0) not in positions
    assert cfg.cylinder.init_state.pos[2] > cfg.table.init_state.pos[2]
    assert cfg.sphere.init_state.pos[2] > cfg.table.init_state.pos[2]
    assert cfg.robot.init_state.joint_pos == {"slider_to_cart": -0.25, "cart_to_pole": 0.45}
    for name in ("ground", "robot", "moving_cube", "table", "cylinder", "sphere"):
        assert ("class", name) in getattr(cfg, name).spawn.semantic_tags


def test_specialized_scenes_own_facts_and_runner_stays_generic() -> None:
    """Scene-specific geometry and tolerances do not leak into the runner."""
    scene_path = _SUITE_DIR / "rendering_scene_cfgs.py"
    classes = _defined_classes(scene_path)
    expected_scenes = {"franka_cloth", "franka_soft", "kuka_heterogeneous", "shadow_hand"}
    assert {
        "RenderingSceneSpec",
        "FrankaClothRenderingSceneCfg",
        "FrankaSoftRenderingSceneCfg",
        "KukaHeterogeneousRenderingSceneCfg",
        "ShadowHandRenderingSceneCfg",
    } <= classes
    assert {case.scene for case in SPECIALIZED_KIT_CASES} == expected_scenes
    assert {case.scene for _, case in SPECIALIZED_KITLESS_CASES} == expected_scenes - {"kuka_heterogeneous"}
    assert KIT_RENDERING_CASES == KIT_CASES + SPECIALIZED_KIT_CASES
    assert KITLESS_RENDERING_CASES == KITLESS_CASES + SPECIALIZED_KITLESS_CASES

    runner_source = (_SUITE_DIR / "rendering_runner.py").read_text()
    assert not [scene for scene in expected_scenes if f'"{scene}"' in runner_source]
    runtime_functions = {
        node.name
        for node in ast.parse((_SUITE_DIR / "rendering_runtime.py").read_text()).body
        if isinstance(node, ast.FunctionDef)
    }
    assert "make_rendering_physics_cfg" in runtime_functions
    assert "make_physics_cfg" not in runtime_functions


def test_reset_and_teardown_policies_keep_their_owners() -> None:
    """Rendering delegates reset policy to MDP and owns only its lifecycle."""
    runtime_tree = ast.parse((_SUITE_DIR / "rendering_runtime.py").read_text())
    rendering_scene = next(
        node for node in runtime_tree.body if isinstance(node, ast.ClassDef) and node.name == "RenderingScene"
    )
    reset = next(node for node in rendering_scene.body if isinstance(node, ast.FunctionDef) and node.name == "reset")
    reset_call = next(
        node
        for node in ast.walk(reset)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "reset_scene_to_default"
    )
    assert {keyword.arg for keyword in reset_call.keywords} == {
        "reset_joint_targets",
        "preserve_fixed_articulation_roots",
    }

    build = next(
        node for node in runtime_tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_rendering_scene"
    )
    assert any(
        ast.unparse(node) == "runtime.scene.close()"
        for try_node in ast.walk(build)
        if isinstance(try_node, ast.Try)
        for node in try_node.finalbody
    )
    assert "def reset_to_default(" not in (_CORE_ROOT / "isaaclab/scene/interactive_scene.py").read_text()
    events_classes = {
        node.name
        for node in ast.parse((_CORE_ROOT / "isaaclab/envs/mdp/events.py").read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "reset_scene_to_default" in events_classes


def test_renderer_matrix_encodes_compatibility_and_cost() -> None:
    """Cases share compatible AOVs and isolate OVRTX's single-AOV calls."""
    assert len(KIT_CASES) <= 10
    assert len(KITLESS_CASES) <= 50
    default_cases = [
        case for case in KIT_CASES if case.variant is None and not set(case.aovs) & set(SIMPLE_SHADING_AOVS)
    ]
    default_cases.extend(
        case for _, case in KITLESS_CASES if case.variant is None and not set(case.aovs) & set(SIMPLE_SHADING_AOVS)
    )
    assert all(len(case.aovs) == 1 for case in default_cases if case.renderer == "ovrtx")
    assert all(len(case.aovs) > 1 for case in default_cases if case.renderer != "ovrtx")
    for physics in ("ovphysx", "newton"):
        actual = tuple(
            case.aovs[0]
            for stage, case in KITLESS_CASES
            if stage == "legacy"
            and case.physics == physics
            and case.renderer == "ovrtx"
            and case.variant is None
            and case.aovs[0] in OVRTX_AOVS
        )
        assert actual == OVRTX_AOVS

    all_cases = [*KIT_RENDERING_CASES, *(case for _, case in KITLESS_RENDERING_CASES)]
    assert all(isinstance(aov, RenderBufferKind) for case in all_cases for aov in case.aovs)
    for case in all_cases:
        assert len(set(case.aovs) & set(SIMPLE_SHADING_AOVS)) <= 1
    motion_cases = [case for case in all_cases if RenderBufferKind.MOTION_VECTORS in case.aovs]
    assert motion_cases
    assert all(
        (RenderBufferKind.MOTION_VECTORS in case.golden_aovs) == (case.physics != "newton") for case in motion_cases
    )


def test_case_identity_names_only_render_dimensions() -> None:
    """Case IDs expose scene/backend/renderer facts and only real variants."""
    assert "profile" not in RenderCase.__dataclass_fields__
    assert not hasattr(RenderCase, "suite")
    case_ids = [case.id for case in KIT_RENDERING_CASES]
    case_ids.extend(f"{stage}-{case.id}" for stage, case in KITLESS_RENDERING_CASES)
    assert len(case_ids) == len(set(case_ids))
    assert not [case_id for case_id in case_ids if any(word in case_id for word in ("canonical", "probe", "standard"))]


def test_golden_inventory_is_derived_from_the_case_matrix() -> None:
    """Checked-in baselines exactly match registered renderer and visualizer cases."""
    expected = {case.scene: set() for case in KIT_RENDERING_CASES}
    expected.update({case.scene: set() for _, case in KITLESS_RENDERING_CASES})
    for case in KIT_RENDERING_CASES:
        expected[case.scene].update(case.golden_filename(aov, "kit") for aov in case.golden_aovs)
    for _, case in KITLESS_RENDERING_CASES:
        expected[case.scene].update(case.golden_filename(aov) for aov in case.golden_aovs)

    renderer_root = _SUITE_DIR / "golden_images/renderers"
    assert {path.name for path in renderer_root.iterdir() if path.is_dir()} == set(expected)
    for scene, filenames in expected.items():
        assert {path.name for path in (renderer_root / scene).glob("*.png")} == filenames
    assert not list(renderer_root.rglob("legacy-*.png"))
    assert not list(renderer_root.rglob("ovstage-*.png"))

    visualizer_expected = {
        f"{physics}-{visualizer}-{mode}.png"
        for physics in ("physx", "newton")
        for visualizer in ("kit", "newton")
        for mode in ("viewport", "tiled")
    }
    assert {path.name for path in (_SUITE_DIR / "golden_images/visualizers").glob("*.png")} == visualizer_expected
