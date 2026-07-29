# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Workflow rendering for OdinV2."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, StrictUndefined

if TYPE_CHECKING:
    from tools.odin.config_file import OdinConfig
    from tools.odin.plan import PlannedRow

__all__ = ["osmo_safe_task_name", "render_workflow_yaml"]


_DNS_1123_LABEL_MAX = 63
_HASH_SUFFIX_LEN = 7  # "-" + 6 hex chars
_NON_ALNUM_DASH = re.compile(r"[^a-z0-9-]")
_RUN_OF_DASHES = re.compile(r"-+")


def osmo_safe_task_name(row_key: str) -> str:
    """Convert an OdinV2 ``row_key`` into a DNS-1123-compliant OSMO task name.

    Constraints, per Kubernetes' DNS-1123 label rules:

    - At most 63 characters.
    - Lowercase alphanumerics and ``-`` only.
    - Must not start or end with ``-``.

    On truncation a six-hex-character hash of the full *row_key* is appended, so
    distinct long inputs still produce distinct outputs.

    Args:
        row_key: Row identity, e.g. ``rsl_rl_physx_Isaac-Ant_seed42``.

    Returns:
        A DNS-1123-safe label.
    """
    dashed = re.sub(r"[_.\s]+", "-", row_key.lower())
    collapsed = _RUN_OF_DASHES.sub("-", _NON_ALNUM_DASH.sub("-", dashed)).strip("-")
    if not collapsed:
        # Degenerate input: emit a stable hash-only label.
        return f"odin-{hashlib.sha256(row_key.encode('utf-8')).hexdigest()[:6]}"
    if len(collapsed) <= _DNS_1123_LABEL_MAX:
        return collapsed
    digest = hashlib.sha256(row_key.encode("utf-8")).hexdigest()[:6]
    return f"{collapsed[: _DNS_1123_LABEL_MAX - _HASH_SUFFIX_LEN].rstrip('-')}-{digest}"


_TEMPLATES_DIR = Path(__file__).parent / "templates"


def render_workflow_yaml(
    *,
    dispatch_id: str,
    chunk_index: int,
    rows: list[PlannedRow],
    cfg: OdinConfig,
    image_ref: str,
    publish_commands: dict[str, str],
) -> str:
    """Render the OSMO workflow YAML for one dispatch chunk.

    Args:
        dispatch_id: Dispatch identifier, ``YYYYMMDD-HHMMSS``.
        chunk_index: Zero-based chunk position; disambiguates workflow names.
        rows: Rows belonging to this chunk.
        cfg: Validated ``odin.yaml`` contents.
        image_ref: Digest-pinned image reference every task in this chunk runs.
        publish_commands: Shell snippet per ``row_key``, spliced into the entry
            script after the benchmark call.

    Returns:
        The rendered workflow YAML.

    Raises:
        KeyError: If a row has no entry in *publish_commands*.
    """
    missing = [row.row_key for row in rows if row.row_key not in publish_commands]
    if missing:
        raise KeyError(f"publish_commands is missing entries for: {missing}")
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    return env.get_template("dispatch.yaml.j2").render(
        dispatch_id=dispatch_id,
        chunk_index=chunk_index,
        rows=rows,
        cfg=cfg,
        image_ref=image_ref,
        publish_commands=publish_commands,
    )
