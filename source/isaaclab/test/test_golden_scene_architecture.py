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
    SIMPLE_SHADING_AOVS,
    select_kitless_cases,
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
    forbidden = ("isaaclab_tasks", "gymnasium", "isaaclab.envs", "ManagerBasedEnv", "DirectRLEnv", "hydra")
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


def test_reset_manager_is_only_an_adapter() -> None:
    """Default-state ownership remains on InteractiveScene, not in manager events."""
    scene_source = (_CORE_PACKAGE / "scene" / "interactive_scene.py").read_text()
    assert scene_source.count("def reset_to_default(") == 1

    events_path = _CORE_PACKAGE / "envs" / "mdp" / "events.py"
    function = next(
        node
        for node in ast.parse(events_path.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "reset_scene_to_default"
    )
    statements = function.body[1:] if ast.get_docstring(function) else function.body
    assert len(statements) == 1 and isinstance(statements[0], ast.Expr)
    call = statements[0].value
    assert isinstance(call, ast.Call) and ast.unparse(call.func) == "env.scene.reset_to_default"
    assert not call.args
    assert {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords} == {
        "env_ids": "env_ids",
        "reset_joint_targets": "reset_joint_targets",
    }


def test_golden_inventory_matches_case_matrix() -> None:
    """Checked-in baselines exactly match the declared renderer and visualizer matrices."""
    renderer_expected = {f"kit-{case.golden_id(aov)}.png" for case in KIT_CASES for aov in case.aovs} | {
        f"{stage}-{case.golden_id(aov)}.png" for stage, case in KITLESS_CASES for aov in case.aovs
    }
    renderer_dir = _RENDERER_TEST_DIR / "golden_images" / "rendering_scene"
    assert {path.name for path in renderer_dir.glob("*.png")} == renderer_expected

    visualizer_expected = {
        f"{physics}-{visualizer}-{mode}.png"
        for physics in ("physx", "newton")
        for visualizer in ("kit", "newton")
        for mode in ("viewport", "tiled")
    }
    visualizer_dir = _VISUALIZER_TEST_DIR / "golden_images" / "rendering_scene"
    assert {path.name for path in visualizer_dir.glob("*.png")} == visualizer_expected
