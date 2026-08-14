# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate a requirements file for resolving Isaac Lab wheels with uv overrides."""

from __future__ import annotations

import argparse
from pathlib import Path

import tomllib


def generate_uv_overrides(root_pyproject: Path, output_path: Path) -> None:
    """Write the root uv dependency overrides as a requirements file.

    Args:
        root_pyproject: Root Isaac Lab ``pyproject.toml`` path.
        output_path: Destination requirements file path.
    """
    with root_pyproject.open("rb") as file:
        root = tomllib.load(file)

    overrides = root.get("tool", {}).get("uv", {}).get("override-dependencies", [])
    if not isinstance(overrides, list) or not all(isinstance(requirement, str) for requirement in overrides):
        raise TypeError("[tool.uv].override-dependencies must be a list of requirement strings")

    output_path.write_text("\n".join(overrides) + "\n", encoding="utf-8")


def main() -> None:
    """Parse command-line arguments and generate the uv override file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_pyproject", type=Path, help="Path to the root pyproject.toml")
    parser.add_argument("output_path", type=Path, help="Path to the generated requirements file")
    args = parser.parse_args()

    generate_uv_overrides(args.root_pyproject, args.output_path)


if __name__ == "__main__":
    main()
