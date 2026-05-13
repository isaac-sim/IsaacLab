import torch

from isaaclab.assets import RigidObject


def reset_robot(scene):
    """Reset robot to a deterministic, non-zero initial joint pose."""

    robot = scene["robot"]

    # Reset scene buffers first. Writing joint/root states after this avoids
    # scene-level reset from restoring old/default states.
    scene.reset()

    # 重置根状态到默认，并按环境原点平移
    root = robot.data.default_root_state.clone()
    root[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])

    # Joint state from asset defaults (used as baseline).
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    # Prefer configured default joint pose from asset/cfg.
    # Only fallback when the arm defaults are effectively all zeros.
    if joint_pos.shape[-1] >= 6:
        arm_default = joint_pos[..., 0:6]
        if torch.allclose(arm_default, torch.zeros_like(arm_default), atol=1e-6):
            joint_pos[..., 0] = 0.0
            joint_pos[..., 1] = -0.2
            joint_pos[..., 2] = 1.4
            joint_pos[..., 3] = -1.6
            joint_pos[..., 4] = -1.5708
            joint_pos[..., 5] = 0.0

    # 写回仿真，并同步位置目标，避免控制器把关节拉回零位。
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)


def reset_fruits(scene):
    """Randomize fruit positions on table after each environment reset."""

    # Keep samples around three anchor points to reduce overlap while
    # still giving noticeable variation for each reset.
    # jitter: half-range in x/y around anchor (m). Lemon uses a larger range for reach diversity.
    fruit_cfg = {
        "apple": {"anchor": (0.90, -0.12), "jitter": (0.12, 0.12), "z": 0.105},
        "lemon": {"anchor": (0.95, 0.00), "jitter": (0.22, 0.22), "z": 0.105},
        "strawberry": {"anchor": (1.00, 0.12), "jitter": (0.12, 0.12), "z": 0.105},
    }

    env_origins = scene.env_origins
    num_envs = env_origins.shape[0]
    device = env_origins.device
    dtype = env_origins.dtype

    for name, cfg in fruit_cfg.items():
        if name not in scene.keys():
            continue
        asset = scene[name]

        # Sample xy around anchors and keep a fixed z above table.
        noise_xy = (torch.rand((num_envs, 2), device=device, dtype=dtype) * 2.0 - 1.0) * torch.tensor(
            cfg["jitter"], device=device, dtype=dtype
        )
        pos_xy = torch.tensor(cfg["anchor"], device=device, dtype=dtype).unsqueeze(0) + noise_xy
        pos_z = torch.full((num_envs, 1), cfg["z"], device=device, dtype=dtype)
        new_pos = torch.cat([pos_xy, pos_z], dim=1) + env_origins

        # RigidObject: pose goes through PhysX (fixes USD/XForm-only updates, e.g. lemon).
        if isinstance(asset, RigidObject):
            root_pose = asset.data.root_link_pose_w.clone()
            root_pose[:, :3] = new_pos
            asset.write_root_pose_to_sim(root_pose)
            asset.write_root_velocity_to_sim(
                torch.zeros((num_envs, 6), device=asset.device, dtype=root_pose.dtype)
            )
            continue

        # Legacy XFormPrim extras: prefer fabric sync for kinematic bodies.
        if hasattr(asset, "get_world_poses") and hasattr(asset, "set_world_poses"):
            try:
                _, orientation = asset.get_world_poses(usd=False)
                asset.set_world_poses(positions=new_pos, orientations=orientation, usd=False)
            except Exception:
                _, orientation = asset.get_world_poses()
                asset.set_world_poses(positions=new_pos, orientations=orientation)
            continue

        if hasattr(asset, "data") and hasattr(asset, "write_root_pose_to_sim"):
            root_state = asset.data.default_root_state.clone()
            root_state[:, :3] = new_pos
            asset.write_root_pose_to_sim(root_state[:, :7])
            if hasattr(asset, "write_root_velocity_to_sim"):
                asset.write_root_velocity_to_sim(root_state[:, 7:])
