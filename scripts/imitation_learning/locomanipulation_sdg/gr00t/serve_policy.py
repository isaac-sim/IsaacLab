# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Serve the G1 policy from the isolated GR00T N1.5 environment."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True, help="Local checkpoint directory or Hugging Face model ID.")
    parser.add_argument("--embodiment_tag", default="new_embodiment")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind; defaults to this machine only.")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    from gr00t.eval.service import BaseInferenceServer
    from policy import Policy

    policy = Policy(args.model_path, args.embodiment_tag)
    server = BaseInferenceServer(host=args.host, port=args.port)
    server.register_endpoint("get_action", policy.policy.get_action)
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.socket.close(linger=0)
        server.context.term()


if __name__ == "__main__":
    main()
