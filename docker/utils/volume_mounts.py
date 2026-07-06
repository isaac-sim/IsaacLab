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

import os
import sys
from pathlib import Path

import yaml

# docker-compose.yaml lives one level up from this ``docker/utils`` package.
DEFAULT_COMPOSE = Path(__file__).resolve().parents[1] / "docker-compose.yaml"

# Top-level compose extension field holding the shared volume list.
_VOLUMES_KEY = "x-default-isaac-lab-volumes"


def named_volume_targets(compose_path: str | os.PathLike = DEFAULT_COMPOSE) -> list[str]:
    """Return the in-container target path of every ``type: volume`` mount.

    Args:
        compose_path: Path to the compose file to read.

    Returns:
        The target paths, with any ``${VAR}`` references left intact.
    """
    data = yaml.safe_load(Path(compose_path).read_text(encoding="utf-8"))
    mounts = data.get(_VOLUMES_KEY) or []
    return [m["target"] for m in mounts if m.get("type") == "volume"]


def resolved_targets(compose_path: str | os.PathLike = DEFAULT_COMPOSE) -> list[str]:
    """Like :func:`named_volume_targets` but with ``${VAR}`` expanded from the environment."""
    return [os.path.expandvars(t).rstrip("/") for t in named_volume_targets(compose_path)]


def main() -> int:
    """Print one resolved mount point per line; fail loudly on an empty or unresolved list."""
    targets = resolved_targets()
    unresolved = [t for t in targets if "$" in t]
    if unresolved:
        print(f"unresolved variables in volume targets: {unresolved}", file=sys.stderr)
        return 1
    if not targets:
        print(f"no named-volume targets found under '{_VOLUMES_KEY}' in docker-compose.yaml", file=sys.stderr)
        return 1
    print("\n".join(targets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
