# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for deterministic training CLI plumbing and seed ordering."""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from packaging.requirements import Requirement
from packaging.version import Version
from tensorboard.backend.event_processing import event_accumulator

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[3]


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
    """Return the skrl lower bound declared by ``source/isaaclab_rl/setup.py``."""
    setup_path = REPO_ROOT / "source/isaaclab_rl/setup.py"
    module = ast.parse(setup_path.read_text(encoding="utf-8"))

    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "EXTRAS_REQUIRE" for target in node.targets):
            continue
        extras_require = ast.literal_eval(node.value)
        for dependency in extras_require["skrl"]:
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

    raise AssertionError("Could not find skrl lower bound in source/isaaclab_rl/setup.py")


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
        "scripts/reinforcement_learning/skrl/play.py",
        "scripts/reinforcement_learning/skrl/play_skrl.py",
        "scripts/reinforcement_learning/skrl/train.py",
        "scripts/reinforcement_learning/skrl/train_skrl.py",
    ]

    for relative_path in skrl_scripts:
        tree = _load_tree(relative_path)
        assert Version(_module_string_constant(tree, "SKRL_VERSION")) == expected_version, relative_path


def test_train_scripts_call_configure_seed_after_runner_or_agent_construction():
    """RL train scripts wire strict PyTorch determinism after runner / agent construction."""
    train_scripts = {
        "scripts/reinforcement_learning/rl_games/train.py": {"Runner"},
        "scripts/reinforcement_learning/skrl/train.py": {"Runner"},
        "scripts/reinforcement_learning/rsl_rl/train.py": {"OnPolicyRunner", "DistillationRunner"},
        "scripts/reinforcement_learning/sb3/train.py": {"PPO"},
    }

    for relative_path, constructors in train_scripts.items():
        tree = _load_tree(relative_path)
        configure_seed_lines = _call_lines(tree, {"configure_seed"})
        constructor_lines = _call_lines(tree, constructors)
        launcher_hook_lines = _call_lines(tree, {"add_launcher_args"})

        assert launcher_hook_lines, f"{relative_path}: expected add_launcher_args(parser) call."
        assert configure_seed_lines, f"{relative_path}: expected configure_seed(...) call."
        assert constructor_lines, f"{relative_path}: expected runner/agent constructor call {constructors}."
        assert min(configure_seed_lines) > max(constructor_lines), (
            f"{relative_path}: configure_seed must be called after runner/agent construction. "
            f"configure_seed lines={configure_seed_lines}, constructor lines={constructor_lines}"
        )


def test_play_scripts_call_configure_seed_after_runner_or_agent_construction():
    """RL play scripts wire strict PyTorch determinism after runner / agent construction."""
    play_scripts = {
        "scripts/reinforcement_learning/rl_games/play.py": {"Runner"},
        "scripts/reinforcement_learning/skrl/play.py": {"Runner"},
        "scripts/reinforcement_learning/rsl_rl/play.py": {"OnPolicyRunner", "DistillationRunner"},
        "scripts/reinforcement_learning/sb3/play.py": {"PPO.load"},
    }

    for relative_path, constructors in play_scripts.items():
        tree = _load_tree(relative_path)
        configure_seed_lines = _call_lines(tree, {"configure_seed"})
        constructor_lines = _call_lines(tree, constructors)
        launcher_hook_lines = _call_lines(tree, {"add_launcher_args"})

        assert launcher_hook_lines, f"{relative_path}: expected add_launcher_args(parser) call."
        assert configure_seed_lines, f"{relative_path}: expected configure_seed(...) call."
        assert constructor_lines, f"{relative_path}: expected runner/agent constructor call {constructors}."
        assert min(configure_seed_lines) > max(constructor_lines), (
            f"{relative_path}: configure_seed must be called after runner/agent construction. "
            f"configure_seed lines={configure_seed_lines}, constructor lines={constructor_lines}"
        )


def _latest_event_file(before: set[Path], logs_root: Path) -> Path:
    candidates = set(logs_root.glob("**/events*"))
    new_files = [p for p in candidates if p not in before]
    if new_files:
        return max(new_files, key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise AssertionError(f"No tensorboard event file was generated under: {logs_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_rewards_per_iter(event_file: Path, preferred_tags: list[str]) -> list[float]:
    ea = event_accumulator.EventAccumulator(str(event_file))
    ea.Reload()
    scalar_tags = ea.Tags()["scalars"]
    selected_tag = None
    for tag in preferred_tags:
        if tag in scalar_tags:
            selected_tag = tag
            break
    if selected_tag is None:
        reward_like_tags = sorted(tag for tag in scalar_tags if "reward" in tag.lower())
        if reward_like_tags:
            selected_tag = reward_like_tags[0]
    if selected_tag is None:
        raise AssertionError(
            f"No reward-like scalar tag found in tensorboard file: {event_file}. Available scalar tags: {scalar_tags}"
        )
    return [event.value for event in ea.Scalars(selected_tag)]


def _run_train_once(
    *,
    train_script: str,
    log_subdir: str,
    preferred_reward_tags: list[str],
    task_name: str,
    deterministic: bool,
) -> list[float]:
    logs_root = REPO_ROOT / "logs" / log_subdir
    before = set(logs_root.glob("**/events*"))
    cmd = [
        "./isaaclab.sh",
        "-p",
        train_script,
        "--task",
        task_name,
        "--enable_cameras",
        "--headless",
        "--seed",
        "42",
        "--max_iterations",
        "50",
    ]
    if deterministic:
        cmd.append("--deterministic")

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=1200,
        check=False,
    )
    assert result.returncode == 0, (
        f"Command failed: {' '.join(cmd)}\n"
        f"--- stdout ---\n{result.stdout[-4000:]}\n"
        f"--- stderr ---\n{result.stderr[-4000:]}\n"
    )
    event_file = _latest_event_file(before, logs_root)
    rewards = _read_rewards_per_iter(event_file, preferred_reward_tags)
    assert rewards, f"No reward series values read from: {event_file}"
    return rewards


def _aligned_rewards(a: list[float], b: list[float]) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    if n == 0:
        raise AssertionError("At least one rewards sequence is empty.")
    return np.asarray(a[:n]), np.asarray(b[:n])


@pytest.mark.skipif(
    os.environ.get("ISAACLAB_RUN_DETERMINISM_TRAIN_TEST", "0") != "1",
    reason="Expensive test: set ISAACLAB_RUN_DETERMINISM_TRAIN_TEST=1 to enable.",
)
def test_rl_games_deterministic_flag_affects_rewards_reproducibility():
    """Non-deterministic runs should diverge; deterministic runs should match (RL-Games tensorboard)."""
    train_script = "scripts/reinforcement_learning/rl_games/train.py"
    log_subdir = "rl_games"
    preferred_reward_tags = ["rewards/iter"]
    task_name = "Isaac-Cartpole-RGB-v0"

    rewards_non_det_1 = _run_train_once(
        train_script=train_script,
        log_subdir=log_subdir,
        preferred_reward_tags=preferred_reward_tags,
        task_name=task_name,
        deterministic=False,
    )
    rewards_non_det_2 = _run_train_once(
        train_script=train_script,
        log_subdir=log_subdir,
        preferred_reward_tags=preferred_reward_tags,
        task_name=task_name,
        deterministic=False,
    )
    rewards_det_1 = _run_train_once(
        train_script=train_script,
        log_subdir=log_subdir,
        preferred_reward_tags=preferred_reward_tags,
        task_name=task_name,
        deterministic=True,
    )
    rewards_det_2 = _run_train_once(
        train_script=train_script,
        log_subdir=log_subdir,
        preferred_reward_tags=preferred_reward_tags,
        task_name=task_name,
        deterministic=True,
    )

    non_det_a, non_det_b = _aligned_rewards(rewards_non_det_1, rewards_non_det_2)
    det_a, det_b = _aligned_rewards(rewards_det_1, rewards_det_2)

    assert not np.allclose(non_det_a, non_det_b, rtol=0.0, atol=1e-6), (
        "Expected non-deterministic runs to produce different rewards/iter curves, but they matched within tolerance."
    )
    assert np.allclose(det_a, det_b, rtol=0.0, atol=1e-6), (
        "Expected deterministic runs to produce matching rewards/iter curves, but they diverged."
    )
