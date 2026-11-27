# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Serve the DROID joint-position checkpoint in OpenPI's isolated environment.

Normally launched automatically by ``pi05_evaluate.py``. For a separate inference machine, run
this file in the pinned OpenPI environment with ``--checkpoint PATH_OR_GS_URL --host 0.0.0.0`` and connect the
evaluation with ``--policy HOST:9184``. Checkpoints and the tokenizer are downloaded on first use.
"""

import argparse
import dataclasses
import logging

from openpi import transforms
from openpi.policies import droid_policy, policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared import download
from openpi.training import config


def main() -> None:
    """Load a cached joint-position policy and expose OpenPI's standard inference protocol."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9184)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # This checkpoint predicts joint deltas, unlike the standard DROID velocity checkpoint.
    # OpenPI unnormalizes them before AbsoluteActions adds the observed joint positions.
    cfg = config.get_config("pi05_droid")
    cfg = dataclasses.replace(
        cfg,
        name="pi05_droid_jointpos",
        data=dataclasses.replace(
            cfg.data,
            data_transforms=lambda model: transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=model.model_type)],
                outputs=[transforms.AbsoluteActions(transforms.make_bool_mask(7, -1)), droid_policy.DroidOutputs()],
            ),
        ),
    )
    checkpoint = download.maybe_download(args.checkpoint, gs={"token": "anon"})
    policy = policy_config.create_trained_policy(cfg, checkpoint)
    WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={"checkpoint": args.checkpoint, "action_space": "absolute_joint_position", "inference": "serial"},
    ).serve_forever()


if __name__ == "__main__":
    main()
