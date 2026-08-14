# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for deterministic training CLI plumbing and seed ordering."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pytest
import tomllib
from packaging.requirements import Requirement
from packaging.version import Version

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_tree(relative_path: str) -> ast.AST:
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    return ast.parse(source)


def _module_string_constant(tree: ast.AST, name: str) -> str:
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return node.value.value

    raise AssertionError(f"Could not find string constant {name}.")


def _source_skrl_min_version() -> Version:
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)

    for dependency in data["project"]["optional-dependencies"]["skrl"]:
        requirement = Requirement(dependency)
        if requirement.name != "skrl":
            continue
        lower_bounds = [
            Version(specifier.version)
            for specifier in requirement.specifier
            if specifier.operator in {">=", "==", "~="}
        ]
        if lower_bounds:
            return max(lower_bounds)

    raise AssertionError("Could not find skrl lower bound in pyproject.toml")


def _called_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        if func.attr == "load" and isinstance(func.value, ast.Name) and func.value.id == "PPO":
            return "PPO.load"
        return func.attr
    return None


def _call_lines(tree: ast.AST, func_names: set[str]) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            called = _called_name(node)
            if called in func_names:
                lines.append(node.lineno)
    return sorted(lines)


def test_app_launcher_adds_deterministic_cli_flag():
    """AppLauncher must expose --deterministic for all train scripts using add_launcher_args."""
    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(["--deterministic"])
    assert hasattr(args, "deterministic")
    assert args.deterministic is True


def test_skrl_scripts_min_version_matches_source_package():
    """skrl runtime guards must match the package metadata lower bound."""
    expected_version = _source_skrl_min_version()
    skrl_scripts = [
        "source/isaaclab_rl/isaaclab_rl/entrypoints/backends/play_skrl.py",
        "source/isaaclab_rl/isaaclab_rl/entrypoints/backends/train_skrl.py",
    ]

    for relative_path in skrl_scripts:
        tree = _load_tree(relative_path)
        assert Version(_module_string_constant(tree, "SKRL_VERSION")) == expected_version, relative_path


def test_unified_scripts_delegate_to_package_entrypoints():
    """Unified executables must remain thin delegates to isaaclab_rl."""
    scripts = (
        ("reinforcement_learning/train.py", "run_train_cli"),
        ("reinforcement_learning/play.py", "run_play_cli"),
        ("environments/zero_agent.py", "run_zero_agent_cli"),
        ("environments/random_agent.py", "run_random_agent_cli"),
    )
    for relative_path, function_name in scripts:
        source = (REPO_ROOT / "scripts" / relative_path).read_text(encoding="utf-8")
        assert f"from isaaclab_rl.entrypoints import {function_name}" in source
        assert f"return {function_name}(argv)" in source


_BACKEND_CONSTRUCTORS = {
    "rl_games": {"Runner"},
    "skrl": {"Runner"},
    "rsl_rl": {"OnPolicyRunner", "DistillationRunner"},
}


@pytest.mark.parametrize(
    ("relative_path", "constructors"),
    [
        *[
            (f"source/isaaclab_rl/isaaclab_rl/entrypoints/backends/train_{backend}.py", constructors)
            for backend, constructors in {**_BACKEND_CONSTRUCTORS, "sb3": {"PPO"}}.items()
        ],
        *[
            (f"source/isaaclab_rl/isaaclab_rl/entrypoints/backends/play_{backend}.py", constructors)
            for backend, constructors in {**_BACKEND_CONSTRUCTORS, "sb3": {"PPO.load"}}.items()
        ],
    ],
)
def test_backends_configure_deterministic_torch_after_runner_or_agent_creation(
    relative_path: str, constructors: set[str]
):
    """Backends must enable deterministic torch operations after runner / agent construction."""
    tree = _load_tree(relative_path)

    configure_seed_lines = _call_lines(tree, {"configure_seed"})
    constructor_lines = _call_lines(tree, constructors)
    launcher_hook_lines = _call_lines(tree, {"add_launcher_args"})

    assert launcher_hook_lines, f"{relative_path}: expected a launcher-argument registration call."
    assert configure_seed_lines, f"{relative_path}: expected configure_seed(...) call."
    assert constructor_lines, f"{relative_path}: expected runner/agent constructor call {constructors}."
    assert min(configure_seed_lines) > max(constructor_lines), (
        f"{relative_path}: configure_seed must be called after runner/agent construction. "
        f"configure_seed lines={configure_seed_lines}, constructor lines={constructor_lines}"
    )
