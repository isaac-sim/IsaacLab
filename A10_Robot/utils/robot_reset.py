import numpy as np


def reset_robot(scene):
    """把双臂机器人 reset 回默认姿态。"""
    root = scene["robot"].data.default_root_state.clone()
    root[:, :3] += scene.env_origins
    scene["robot"].write_root_pose_to_sim(root[:, :7])
    scene["robot"].write_root_velocity_to_sim(root[:, 7:])

    joint_pos = scene["robot"].data.default_joint_pos.clone()
    joint_vel = scene["robot"].data.default_joint_vel.clone()
    scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)

    scene.reset()
