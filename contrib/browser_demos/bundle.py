# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pack runtime binaries and reviewed RSL-RL actors for the shared browser widget."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def write_binary_files(data: bytes, output: Path, filename: str) -> list[str]:
    """Write binary parts below the static preview hosts' per-file limit.

    Args:
        data: Complete binary payload, concatenated in returned filename order.
        output: Directory receiving the files.
        filename: Filename for an unsplit payload.

    Returns:
        Filenames in payload order.
    """
    limit = 1_900_000
    path = Path(filename)
    files = (
        [filename]
        if len(data) <= limit
        else [f"{path.stem}-{i}{path.suffix}" for i in range((len(data) + limit - 1) // limit)]
    )
    for i, name in enumerate(files):
        (output / name).write_bytes(data[i * limit : (i + 1) * limit])
    return files


def write_policy(checkpoint: Path, output: Path, widths: tuple[int, ...]) -> dict[str, object]:
    """Write actor weights after checking the expected layer widths.

    Observation normalization belongs to the captured observation step.

    Args:
        checkpoint: Published RSL-RL checkpoint.
        output: Directory receiving the weight file.
        widths: Expected input, hidden, and output widths.
    """
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)["actor_state_dict"]
    shapes = tuple(zip(widths[1:], widths[:-1], strict=True))
    chunks, layers, offset = [], [], 0
    for layer, (rows, columns) in enumerate(shapes):
        weight = weights[f"mlp.{layer * 2}.weight"].detach().numpy().astype("<f4", copy=False)
        bias = weights[f"mlp.{layer * 2}.bias"].detach().numpy().astype("<f4", copy=False)
        if weight.shape != (rows, columns) or bias.shape != (rows,):
            raise ValueError(f"Unexpected checkpoint layer {layer}: {weight.shape}, {bias.shape}")
        chunks.extend((weight.tobytes(), bias.tobytes()))
        layers.append({"rows": rows, "columns": columns, "offset": offset})
        offset += rows * (columns + 1)
    packed = b"".join(chunks)
    files = write_binary_files(packed, output, "policy.bin")
    source = {"file": files[0]} if len(files) == 1 else {"files": files}
    return {**source, "layers": layers, "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
