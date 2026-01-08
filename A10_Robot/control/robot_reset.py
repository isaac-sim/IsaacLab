import numpy as np


def reset_robot(scene):
    """将机器人重置到默认状态，并在仅12关节（无夹爪）场景下屏蔽夹爪位。

    约定：
    - 若关节为 14 维，索引 6 和 13 为左右夹爪，重置时置为 0。
    - 若关节为 12 维（无夹爪），直接使用默认关节位姿/速度。
    """

    robot = scene["robot"]

    # 重置根状态到默认，并按环境原点平移
    root = robot.data.default_root_state.clone()
    root[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])

    # 关节状态
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    # 若为 14 维，屏蔽夹爪通道（索引 6 与 13）
    if hasattr(joint_pos, "numel") and joint_pos.shape[-1] == 14:
        joint_pos[..., 6] = 0.0
        joint_pos[..., 13] = 0.0
        if hasattr(joint_vel, "numel") and joint_vel.shape[-1] == 14:
            joint_vel[..., 6] = 0.0
            joint_vel[..., 13] = 0.0

    # 写回仿真
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    # 让场景组件（传感器等）同步重置
    scene.reset()
