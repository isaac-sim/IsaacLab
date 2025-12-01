import numpy as np


def reset_robot(scene):
    """将双臂机器人重置为默认状态。

    约定：若模型是 14 维（双七轴含夹爪），则将夹爪索引 6 与 13 置 0；
    若模型是 12 维（双六轴不含夹爪），直接恢复默认关节状态。
    """

    robot = scene["robot"]

    # 根状态重置（并加上环境原点偏移）
    root = robot.data.default_root_state.clone()
    root[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])

    # 关节状态重置
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    # 若为 14 维，清零夹爪位
    if hasattr(joint_pos, "shape") and joint_pos.shape[-1] == 14:
        joint_pos[..., 6] = 0.0
        joint_pos[..., 13] = 0.0
        if hasattr(joint_vel, "shape") and joint_vel.shape[-1] == 14:
            joint_vel[..., 6] = 0.0
            joint_vel[..., 13] = 0.0

    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    # 场景组件同步重置（传感器等）
    scene.reset()
