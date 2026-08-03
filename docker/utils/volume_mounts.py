# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve the named-volume mount points declared in ``docker-compose.yaml``.

``docker-compose.yaml`` is the single source of truth for the directories that
persist across containers. The non-root image build calls this module to
pre-create and ``chown`` each named-volume mount point to the runtime user, and
the regression test calls it to validate that parsing works. Keeping one parser
avoids a hand-maintained second list in the Dockerfiles drifting from compose.

A fresh Docker named volume inherits ownership from the image directory at its
mount path on first mount; if that directory is missing or root-owned, the
volume comes up root-owned and the non-root ``isaaclab`` user cannot write it.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import yaml

# docker-compose.yaml lives one level up from this ``docker/utils`` package.
DEFAULT_COMPOSE = Path(__file__).resolve().parents[1] / "docker-compose.yaml"

# Top-level Compose extension field holding the default shared volume list.
_DEFAULT_VOLUMES_KEY = "x-default-isaac-lab-volumes"
_COMPOSE_DEFAULT_VARIABLE_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")


def _expand_compose_variables(value: str) -> str:
    """Expand environment variables, including Compose ``${VAR:-default}`` expressions."""

    def replace_default(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1)) or match.group(2)

    return os.path.expandvars(_COMPOSE_DEFAULT_VARIABLE_PATTERN.sub(replace_default, value))


def named_volume_targets(
    compose_path: str | os.PathLike[str] = DEFAULT_COMPOSE, volumes_key: str = _DEFAULT_VOLUMES_KEY
) -> list[str]:
    """Return the in-container target path of every ``type: volume`` mount.

    Args:
        compose_path: Path to the compose file to read.
        volumes_key: Top-level Compose extension key containing the mount list.

    Returns:
        The target paths, with any ``${VAR}`` references left intact.
    """
    data = yaml.safe_load(Path(compose_path).read_text(encoding="utf-8"))
    mounts = data.get(volumes_key) or []
    return [m["target"] for m in mounts if m.get("type") == "volume"]


def resolved_targets(
    compose_path: str | os.PathLike[str] = DEFAULT_COMPOSE, volumes_key: str = _DEFAULT_VOLUMES_KEY
) -> list[str]:
    """Like :func:`named_volume_targets` but with Compose variables expanded from the environment."""
    return [_expand_compose_variables(t).rstrip("/") for t in named_volume_targets(compose_path, volumes_key)]


def main() -> int:
    """Print one resolved mount point per line; fail loudly on an empty or unresolved list."""
    parser = argparse.ArgumentParser(description="Resolve named-volume targets from docker-compose.yaml.")
    parser.add_argument(
        "--volumes_key",
        default=_DEFAULT_VOLUMES_KEY,
        help="Top-level Compose extension key containing the mount list.",
    )
    args = parser.parse_args()

    targets = resolved_targets(volumes_key=args.volumes_key)
    unresolved = [t for t in targets if "$" in t]
    if unresolved:
        print(f"unresolved variables in volume targets: {unresolved}", file=sys.stderr)
        return 1
    if not targets:
        print(f"no named-volume targets found under '{args.volumes_key}' in docker-compose.yaml", file=sys.stderr)
        return 1
    print("\n".join(targets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
