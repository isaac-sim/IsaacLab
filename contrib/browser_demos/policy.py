# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pack reviewed RSL-RL MLP actors for the shared browser evaluator."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


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
    # Keep exact float32 weights while staying below static preview hosts' per-file limit.
    limit = 1_900_000
    files = (
        ["policy.bin"]
        if len(packed) <= limit
        else [f"policy-{i}.bin" for i in range((len(packed) + limit - 1) // limit)]
    )
    for i, filename in enumerate(files):
        (output / filename).write_bytes(packed[i * limit : (i + 1) * limit])
    source = {"file": files[0]} if len(files) == 1 else {"files": files}
    return {**source, "layers": layers, "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
