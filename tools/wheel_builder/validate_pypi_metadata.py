# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reject wheel dependency metadata that PyPI will not accept."""

from __future__ import annotations

import argparse
import email
from pathlib import Path
from zipfile import ZipFile

from packaging.requirements import Requirement


def validate_pypi_metadata(wheel_path: Path) -> None:
    """Raise an error when a wheel contains direct URL dependencies."""
    with ZipFile(wheel_path) as wheel:
        metadata_paths = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError(f"Expected one METADATA file in {wheel_path}, found {len(metadata_paths)}")
        metadata = email.message_from_bytes(wheel.read(metadata_paths[0]))

    direct_urls = [requirement for requirement in metadata.get_all("Requires-Dist", []) if Requirement(requirement).url]
    if direct_urls:
        formatted = "\n".join(f"  - {requirement}" for requirement in direct_urls)
        raise ValueError(f"PyPI does not allow direct URL dependencies in {wheel_path}:\n{formatted}")


def main() -> None:
    """Validate wheel paths provided on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    for wheel_path in args.wheels:
        try:
            validate_pypi_metadata(wheel_path)
        except ValueError as error:
            errors.append(str(error))
        else:
            print(f"PyPI metadata validation passed: {wheel_path}")
    if errors:
        raise SystemExit("\n\n".join(errors))


if __name__ == "__main__":
    main()
