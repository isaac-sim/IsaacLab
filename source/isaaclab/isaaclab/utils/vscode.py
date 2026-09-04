# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for generating Pyright import paths for Isaac Lab projects."""

import importlib.metadata
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys


def resolve_isaacsim_dir(project_dir: pathlib.Path, isaac_path: str | None = None) -> pathlib.Path | None:
    """Resolve the Isaac Sim installation directory.

    Args:
        project_dir: Project root containing an optional ``_isaac_sim`` link.
        isaac_path: Explicit Isaac Sim path, or None to discover the installation.

    Returns:
        The resolved installation directory, or None if Isaac Sim is unavailable.

    Raises:
        ValueError: If an explicit path does not identify a directory.
    """
    if isaac_path:
        explicit_path = pathlib.Path(isaac_path).expanduser()
        if not explicit_path.is_dir():
            raise ValueError(f"Isaac Sim directory does not exist: {explicit_path}")
        return explicit_path.resolve()

    env_path = os.environ.get("ISAAC_PATH")
    if env_path and pathlib.Path(env_path).is_dir():
        return pathlib.Path(env_path).resolve()

    probe = subprocess.run(
        [sys.executable, "-c", "import isaacsim, os; print(os.environ.get('ISAAC_PATH', ''))"],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    for line in reversed(probe.stdout.splitlines()):
        candidate = pathlib.Path(line.strip()).expanduser()
        if line.strip() and candidate.is_dir():
            return candidate.resolve()

    fallback = project_dir / "_isaac_sim"
    return fallback.resolve() if fallback.is_dir() else None


def read_isaacsim_extra_paths(isaacsim_dir: pathlib.Path | None) -> list[pathlib.Path]:
    """Read Isaac Sim's Python extension paths.

    Args:
        isaacsim_dir: Isaac Sim installation directory, or None.

    Returns:
        Absolute extension search paths.
    """
    if isaacsim_dir is None:
        print("[WARN] Isaac Sim was not found; simulator extension paths were not added.")
        return []

    settings_file = isaacsim_dir / ".vscode" / "settings.json"
    if not settings_file.is_file():
        print(f"[WARN] Isaac Sim VS Code settings were not found: {settings_file}")
        return []

    settings = settings_file.read_text(encoding="utf-8")
    match = re.search(r'"python\.analysis\.extraPaths"\s*:\s*\[(.*?)\]', settings, flags=re.DOTALL)
    if match is None:
        print(f"[WARN] python.analysis.extraPaths was not found in {settings_file}")
        return []

    paths = []
    for encoded_path in re.findall(r'"((?:\\.|[^"\\])*)"', match.group(1)):
        path = pathlib.Path(json.loads(f'"{encoded_path}"'))
        paths.append(path if path.is_absolute() else isaacsim_dir / path)
    return paths


def find_isaaclab_package_paths() -> list[pathlib.Path]:
    """Find Isaac Lab package roots visible to the active interpreter.

    Returns:
        Import roots for installed Isaac Lab packages.
    """
    paths = []
    package_names = sorted(name for name in importlib.metadata.packages_distributions() if name.startswith("isaaclab"))
    for package_name in package_names:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            continue
        if spec.submodule_search_locations:
            paths.extend(pathlib.Path(location).parent for location in spec.submodule_search_locations)
        elif spec.origin:
            paths.append(pathlib.Path(spec.origin).parent)
    return paths


def build_extra_paths(project_dir: pathlib.Path, isaacsim_dir: pathlib.Path | None) -> list[str]:
    """Build Pyright search paths for simulator, local, and installed packages.

    Args:
        project_dir: Project root used to discover local source packages.
        isaacsim_dir: Isaac Sim installation directory, or None.

    Returns:
        Deduplicated paths, relative to the project where practical.
    """
    paths = read_isaacsim_extra_paths(isaacsim_dir)
    source_dir = project_dir / "source"
    if source_dir.is_dir():
        paths.extend(path for path in sorted(source_dir.iterdir()) if path.is_dir())
    paths.extend(find_isaaclab_package_paths())

    formatted_paths = []
    seen = set()
    resolved_project_dir = project_dir.resolve()
    for path in paths:
        resolved_path = path.resolve()
        try:
            formatted_path = resolved_path.relative_to(resolved_project_dir).as_posix()
        except ValueError:
            formatted_path = resolved_path.as_posix()
        if formatted_path not in seen:
            seen.add(formatted_path)
            formatted_paths.append(formatted_path)
    return formatted_paths


def write_pyright_config(project_dir: pathlib.Path, extra_paths: list[str]):
    """Write a machine-local Pyright configuration that preserves project policy.

    Args:
        project_dir: Project root where the configuration is written.
        extra_paths: Additional import search paths.
    """
    config: dict[str, object] = {"extraPaths": extra_paths}
    if (project_dir / "pyproject.toml").is_file():
        config["extends"] = "./pyproject.toml"
    (project_dir / "pyrightconfig.json").write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
