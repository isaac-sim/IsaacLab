# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cosmos prompts read from text and JSON files."""

from __future__ import annotations

import glob
import json
from collections.abc import Sequence
from pathlib import Path


def read_prompt_files(paths: Sequence[str]) -> list[str]:
    """Read one Cosmos prompt per file, in the given order.

    A ``.txt`` file is the prompt, without leading and trailing whitespace. A ``.json`` file, such as a structured
    caption, keeps its structure: the parsed object is serialized once into the prompt string, the format Cosmos was
    trained on, rather than flattened into prose. A path with ``*``, ``?``, or ``[`` is a pattern that expands to its
    matches in sorted order.

    Args:
        paths: File paths or patterns.

    Returns:
        The prompts, one per file. A list cycles per episode with one environment and assigns one prompt per
        environment with several; see :attr:`~isaaclab_experimental.cosmos.CosmosModelCfg.prompt`.

    Raises:
        ValueError: If no file is given, a pattern matches nothing, or a file is missing, empty, neither ``.txt`` nor
            ``.json``, or invalid JSON.
    """
    files = []
    for path in (path.strip() for path in paths):
        if any(character in path for character in "*?["):
            matches = sorted(glob.glob(str(Path(path).expanduser())))
            if not matches:
                raise ValueError(f"No Cosmos prompt file matches {path!r}.")
            files.extend(Path(match) for match in matches)
        elif path:
            files.append(Path(path).expanduser())
    if not files:
        raise ValueError("Cosmos prompt files need at least one path.")
    return [_read_prompt_file(path) for path in files]


def _read_prompt_file(path: Path) -> str:
    if path.suffix.lower() not in (".txt", ".json"):
        raise ValueError(f"Cosmos prompt file {path} must be a .txt or .json file.")
    if not path.is_file():
        raise ValueError(f"Cosmos prompt file {path} does not exist.")
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Cosmos prompt file {path} is empty.")
    if path.suffix.lower() == ".json":
        try:
            caption = json.loads(prompt)
        except json.JSONDecodeError as error:
            raise ValueError(f"Cosmos prompt file {path} is not valid JSON: {error}.") from error
        # One serialization keeps keys, order, and text; only the file's layout whitespace changes.
        return json.dumps(caption, ensure_ascii=False)
    return prompt
