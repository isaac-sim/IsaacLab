# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage the aggregate Isaac Lab Python package."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def stage_package(repo_root: Path, stage_dir: Path, version: str) -> None:
    """Create the aggregate package source tree consumed by the wheel builder."""
    builder_dir = repo_root / "tools" / "wheel_builder"
    package_dir = stage_dir / "src" / "isaaclab"

    shutil.rmtree(stage_dir, ignore_errors=True)
    package_dir.mkdir(parents=True)

    shutil.copytree(repo_root / "apps", package_dir / "apps")
    shutil.copytree(repo_root / "source", package_dir / "source")
    shutil.copytree(repo_root / "tools" / "template", package_dir / "tools" / "template")

    for directory in (package_dir / "apps").rglob("*"):
        if directory.is_dir():
            (directory / "__init__.py").touch()

    core_package = package_dir / "source" / "isaaclab" / "isaaclab"
    shutil.copytree(core_package, package_dir, dirs_exist_ok=True)
    shutil.rmtree(core_package)

    for extension_dir in sorted((package_dir / "source").glob("isaaclab_*")):
        package_name = extension_dir.name
        inner_package = extension_dir / package_name
        if not (inner_package / "__init__.py").is_file():
            continue

        installed_package = stage_dir / "src" / package_name
        shutil.copytree(inner_package, installed_package)
        for resource_name in ("config", "data"):
            resource_dir = extension_dir / resource_name
            if resource_dir.is_dir():
                shutil.copytree(resource_dir, installed_package / resource_name)

        init_path = installed_package / "__init__.py"
        init_contents = init_path.read_text(encoding="utf-8")
        init_contents = init_contents.replace(
            'os.path.join(os.path.dirname(__file__), "../"',
            'os.path.join(os.path.dirname(__file__), ""',
        )
        init_path.write_text(init_contents, encoding="utf-8")

        shutil.rmtree(inner_package)
        shutil.rmtree(extension_dir / "data", ignore_errors=True)

    for cache_dir in sorted(stage_dir.rglob("__pycache__"), reverse=True):
        shutil.rmtree(cache_dir, ignore_errors=True)
    for egg_info_dir in sorted(stage_dir.rglob("*.egg-info"), reverse=True):
        shutil.rmtree(egg_info_dir, ignore_errors=True)
    for bytecode_file in stage_dir.rglob("*.pyc"):
        bytecode_file.unlink()

    shutil.copy2(builder_dir / "res" / "__main__.py", package_dir / "__main__.py")
    subprocess.run(
        [
            sys.executable,
            str(builder_dir / "gen_pyproject.py"),
            str(repo_root / "pyproject.toml"),
            str(stage_dir / "pyproject.toml"),
            version,
        ],
        check=True,
    )


def main() -> None:
    """Stage an aggregate package from the containing Isaac Lab checkout."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage_dir", type=Path)
    parser.add_argument("version")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    stage_package(repo_root, args.stage_dir.resolve(), args.version)


if __name__ == "__main__":
    main()
