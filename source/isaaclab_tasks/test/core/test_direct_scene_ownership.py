# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Architecture gates for declarative direct-task scenes."""

import ast
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT / "tools/template"))

import generator  # noqa: E402

sys.path.pop(0)

_DIRECT_ROOTS = (
    Path(__file__).resolve().parents[2] / "isaaclab_tasks",
    _REPO_ROOT / "source/isaaclab_tasks_experimental/isaaclab_tasks_experimental",
    _REPO_ROOT / "scripts",
)
_DIRECT_BASES = {"DirectRLEnv", "DirectMARLEnv", "DirectRLEnvWarp"}
_FORBIDDEN_CALLS = {"clone_plan_from_env_0", "ReplicateSession", "replicate", "VisualizationMarkers"}


def _name(node: ast.expr) -> str:
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""


def test_direct_tasks_leave_scene_construction_to_their_cfg(tmp_path) -> None:
    """Maintained Direct tasks must leave scene construction and cloning to their scene cfg."""
    offenders = []
    modules = []
    for root in _DIRECT_ROOTS:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            modules.append((path, tree))

    direct_types = set(_DIRECT_BASES)
    while (
        descendants := {
            cls.name
            for _, tree in modules
            for cls in tree.body
            if isinstance(cls, ast.ClassDef) and {_name(base) for base in cls.bases} & direct_types
        }
        - direct_types
    ):
        direct_types.update(descendants)

    for path, tree in modules:
        for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
            if not {_name(base) for base in cls.bases} & direct_types:
                continue
            for node in ast.walk(cls):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_setup_scene":
                    offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: _setup_scene")
                if isinstance(node, ast.Call):
                    call = _name(node.func)
                    manual_spawn = (
                        call == "func" and isinstance(node.func, ast.Attribute) and _name(node.func.value) == "spawn"
                    )
                    if call in _FORBIDDEN_CALLS or manual_spawn:
                        offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: {call}")

    for initial_content in ("cartpole", "stubbed"):
        generated = generator._generate_tasks(
            {
                "external": True,
                "initial_content": initial_content,
                "name": "test",
                "workflows": [
                    {"name": "direct", "type": "single-agent"},
                    {"name": "direct", "type": "multi-agent"},
                ],
                "rl_libraries": [],
            },
            str(tmp_path / initial_content),
        )
        for specification in generated:
            task = specification["task"]
            path = Path(task["dir"]) / f"{task['env_filename']}.py"
            source = path.read_text(encoding="utf-8")
            offenders.extend(
                f"generated/{path.relative_to(tmp_path)}: {pattern}"
                for pattern in ("_setup_scene", "clone_plan_from_env_0", "ReplicateSession", "cloner.replicate")
                if pattern in source
            )

    assert not offenders, "Direct tasks own scene construction or cloning:\n" + "\n".join(offenders)
