# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a native VelStand candidate with its normalizer and embedded weights, without a simulator."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from rsl_rl.utils import resolve_class
from tensordict import TensorDict

from isaaclab_tasks.contrib.microduck.velstand.agents.ppo import CONTRACT
from isaaclab_tasks.contrib.microduck.velstand.agents.rsl_rl_ppo_cfg import MicroDuckVelStandPPORunnerCfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("microduck_velstand_state")
    if state is None or state.get("contract") != CONTRACT:
        raise ValueError("Expected a native candidate with the current VelStand contract")
    actor_class, actor_cfg = resolve_class(MicroDuckVelStandPPORunnerCfg().actor.to_dict())
    obs = TensorDict({"policy": torch.zeros(1, 61)}, batch_size=[1])
    actor = actor_class(obs, {"actor": ["policy"]}, "actor", 14, **actor_cfg).eval()
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    exported = actor.as_onnx(verbose=False).cpu().eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Use the native RSL-RL normalizer-inclusive export wrapper, with one portable graph file.
    torch.onnx.export(
        exported,
        exported.get_dummy_inputs(),
        str(args.output),
        export_params=True,
        opset_version=18,
        input_names=exported.input_names,
        output_names=exported.output_names,
        external_data=False,
    )
    with args.checkpoint.open("rb") as stream:
        source_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    with args.output.open("rb") as stream:
        output_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": source_sha,
        "onnx": str(args.output),
        "onnx_sha256": output_sha,
        "input": [1, 61],
        "output": [1, 14],
        "dtype": "float32",
        "normalizer_included": True,
        "normalizer_epsilon": actor.obs_normalizer.eps,
        "normalizer_count": int(actor.obs_normalizer.count),
        "iteration": checkpoint["iter"],
        "microduck_velstand_state": state,
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
