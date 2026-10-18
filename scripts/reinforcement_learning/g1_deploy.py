# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a LEAPP-exported policy on a physical Unitree G1.

Inference is the official ``leapp.InferenceManager``. This file supplies the two ends the graph
cannot reach by itself: the robot's state and camera on the way in, and its motors on the way out.
No simulator is launched and Isaac Sim is never imported.
"""

from __future__ import annotations

import argparse

from g1_hardware import DepthStream, G1Link, Teleop, bring_up

CONTROL_DT = 0.02
"""Control period [s]. The exported graph carries its history at this rate."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pipeline", required=True, help="Exported LEAPP .yaml.")
    parser.add_argument("--interface", required=True, help="Network interface on the robot's subnet.")
    parser.add_argument("--domain_id", type=int, default=0, help="DDS domain.")
    parser.add_argument("--depth_port", type=int, default=5601, help="UDP port the publisher sends to.")
    args = parser.parse_args()

    from leapp import InferenceManager

    inference = InferenceManager(args.pipeline)
    link = G1Link(args.domain_id, args.interface)
    depth = DepthStream(args.depth_port)
    teleop = Teleop()

    bring_up(link, inference.default_joint_pos, ramp_s=3.0)
    inference.reset()

    with link.commanding():
        while teleop.running:
            state = link.read()
            outputs = inference.run_policy({
                "robot_joint_pos": state.joint_pos,
                "robot_joint_vel": state.joint_vel,
                "robot_root_quat_w": state.root_quat_xyzw,
                "robot_root_ang_vel_b": state.ang_vel,
                "base_velocity": teleop.command(state.root_quat_xyzw),
                "depth_camera_output_distance_to_image_plane": depth.latest(),
            })
            link.write(
                joint_pos=outputs["joint_pos"],
                kp=outputs["joint_pos_kp_gains"],
                kd=outputs["joint_pos_kd_gains"],
            )
            link.wait(CONTROL_DT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
