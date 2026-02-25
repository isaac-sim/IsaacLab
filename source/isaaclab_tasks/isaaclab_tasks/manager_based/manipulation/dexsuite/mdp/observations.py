# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import warp as wp

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul, subtract_frame_transforms

from .utils import sample_object_point_cloud

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import TiledCamera


def object_pos_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Object position in the robot's root frame.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot (reference frame). Defaults to ``SceneEntityCfg("robot")``.
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.

    Returns:
        Tensor of shape ``(num_envs, 3)``: object position [x, y, z] expressed in the robot root frame.
    """
    robot = env.scene[robot_cfg.name]
    object = env.scene[object_cfg.name]
    robot_root_pos_w = wp.to_torch(robot.data.root_pos_w)
    object_root_pos_w = wp.to_torch(object.data.root_pos_w)
    robot_root_quat_w = wp.to_torch(robot.data.root_quat_w)
    return quat_apply_inverse(robot_root_quat_w, object_root_pos_w - robot_root_pos_w)


def object_quat_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object orientation in the robot's root frame.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot (reference frame). Defaults to ``SceneEntityCfg("robot")``.
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.

    Returns:
        Tensor of shape ``(num_envs, 4)``: object quaternion ``(w, x, y, z)`` in the robot root frame.
    """
    robot = env.scene[robot_cfg.name]
    object = env.scene[object_cfg.name]
    return quat_mul(quat_inv(wp.to_torch(robot.data.root_quat_w)), wp.to_torch(object.data.root_quat_w))


def body_state_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Body state (pos, quat, lin vel, ang vel) in the base asset's root frame.

    The state for each body is stacked horizontally as
    ``[position(3), quaternion(4)(wxyz), linvel(3), angvel(3)]`` and then concatenated over bodies.

    Args:
        env: The environment.
        body_asset_cfg: Scene entity for the articulated body whose links are observed.
        base_asset_cfg: Scene entity providing the reference (root) frame.

    Returns:
        Tensor of shape ``(num_envs, num_bodies * 13)`` with per-body states expressed in the base root frame.
    """
    body_asset: Articulation = env.scene[body_asset_cfg.name]
    base_asset: Articulation = env.scene[base_asset_cfg.name]
    # get world pose of bodies
    body_pos_w = wp.to_torch(body_asset.data.body_pos_w)[:, body_asset_cfg.body_ids].view(-1, 3)
    body_quat_w = wp.to_torch(body_asset.data.body_quat_w)[:, body_asset_cfg.body_ids].view(-1, 4)
    body_lin_vel_w = wp.to_torch(body_asset.data.body_lin_vel_w)[:, body_asset_cfg.body_ids].view(-1, 3)
    body_ang_vel_w = wp.to_torch(body_asset.data.body_ang_vel_w)[:, body_asset_cfg.body_ids].view(-1, 3)
    num_bodies = int(body_pos_w.shape[0] / env.num_envs)
    # get world pose of base frame
    root_pos_w = (
        wp.to_torch(base_asset.data.root_link_pos_w).unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 3)
    )
    root_quat_w = (
        wp.to_torch(base_asset.data.root_link_quat_w).unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 4)
    )
    # transform from world body pose to local body pose
    body_pos_b, body_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    body_lin_vel_b = quat_apply_inverse(root_quat_w, body_lin_vel_w)
    body_ang_vel_b = quat_apply_inverse(root_quat_w, body_ang_vel_w)
    # concate and return
    out = torch.cat((body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b), dim=1)
    return out.view(env.num_envs, -1)


class object_point_cloud_b(ManagerTermBase):
    """Object surface point cloud expressed in a reference asset's root frame.

    Points are pre-sampled on the object's surface in its local frame and transformed to world,
    then into the reference (e.g., robot) root frame. Optionally visualizes the points.

    Args (from ``cfg.params``):
        object_cfg: Scene entity for the object to sample. Defaults to ``SceneEntityCfg("object")``.
        ref_asset_cfg: Scene entity providing the reference frame. Defaults to ``SceneEntityCfg("robot")``.
        num_points: Number of points to sample on the object surface. Defaults to ``10``.
        visualize: Whether to draw markers for the points. Defaults to ``True``.
        static: If ``True``, cache world-space points on reset and reuse them (no per-step resampling).

    Returns (from ``__call__``):
        If ``flatten=False``: tensor of shape ``(num_envs, num_points, 3)``.
        If ``flatten=True``: tensor of shape ``(num_envs, 3 * num_points)``.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self.ref_asset_cfg: SceneEntityCfg = cfg.params.get("ref_asset_cfg", SceneEntityCfg("robot"))
        num_points: int = cfg.params.get("num_points", 10)
        self.object = env.scene[self.object_cfg.name]
        self.ref_asset: Articulation = env.scene[self.ref_asset_cfg.name]
        # lazy initialize visualizer and point cloud
        if cfg.params.get("visualize", True):
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

            ray_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/ObservationPointCloud")
            ray_cfg.markers["hit"].radius = 0.0025
            self.visualizer = VisualizationMarkers(ray_cfg)
        self.points_local = sample_object_point_cloud(
            env.num_envs, num_points, self.object.cfg.prim_path, device=env.device
        )
        self.points_w = torch.zeros_like(self.points_local)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ref_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        num_points: int = 10,
        flatten: bool = False,
        visualize: bool = True,
    ):
        """Compute the object point cloud in the reference asset's root frame.

        Note:
            Points are pre-sampled at initialization using ``self.num_points``; the ``num_points`` argument is
            kept for API symmetry and does not change the sampled set at runtime.

        Args:
            env: The environment.
            ref_asset_cfg: Reference frame provider (root). Defaults to ``SceneEntityCfg("robot")``.
            object_cfg: Object to sample. Defaults to ``SceneEntityCfg("object")``.
            num_points: Unused at runtime; see note above.
            flatten: If ``True``, return a flattened tensor ``(num_envs, 3 * num_points)``.
            visualize: If ``True``, draw markers for the points.

        Returns:
            Tensor of shape ``(num_envs, num_points, 3)`` or flattened if requested.
        """
        ref_pos_w = wp.to_torch(self.ref_asset.data.root_pos_w).unsqueeze(1).repeat(1, num_points, 1)
        ref_quat_w = wp.to_torch(self.ref_asset.data.root_quat_w).unsqueeze(1).repeat(1, num_points, 1)

        object_pos_w = wp.to_torch(self.object.data.root_pos_w).unsqueeze(1).repeat(1, num_points, 1)
        object_quat_w = wp.to_torch(self.object.data.root_quat_w).unsqueeze(1).repeat(1, num_points, 1)
        # apply rotation + translation
        self.points_w = quat_apply(object_quat_w, self.points_local) + object_pos_w
        if visualize:
            self.visualizer.visualize(translations=self.points_w.view(-1, 3))
        object_point_cloud_pos_b, _ = subtract_frame_transforms(ref_pos_w, ref_quat_w, self.points_w, None)

        return object_point_cloud_pos_b.view(env.num_envs, -1) if flatten else object_point_cloud_pos_b


def fingers_contact_force_b(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """base-frame contact forces from listed sensors, concatenated per env.

    Args:
        env: The environment.
        contact_sensor_names: Names of contact sensors in ``env.scene.sensors`` to read.

    Returns:
        Tensor of shape ``(num_envs, 3 * num_sensors)`` with forces stacked horizontally as
        ``[fx, fy, fz]`` per sensor.
    """
    # Contact sensor returns Warp arrays: force_matrix_w is (N, S, F) vec3f -> torch (N, S, F, 3)
    force_w = [
        wp.to_torch(env.scene.sensors[name].data.force_matrix_w).sum(dim=(1, 2)) for name in contact_sensor_names
    ]
    force_w = torch.stack(force_w, dim=1)
    robot: Articulation = env.scene[asset_cfg.name]
    forces_b = quat_apply_inverse(
        wp.to_torch(robot.data.root_link_quat_w).unsqueeze(1).repeat(1, force_w.shape[1], 1), force_w
    )
    return forces_b.view(env.num_envs, -1)


class vision_camera(ManagerTermBase):

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params.get("sensor_cfg", SceneEntityCfg("tiled_camera"))
        self.sensor: TiledCamera = env.scene.sensors[sensor_cfg.name]
        self.sensor_type = self.sensor.cfg.data_types[0]
        self.norm_fn = self._depth_norm if self.sensor_type == "distance_to_image_plane" else self._rgb_norm

    def __call__(
        self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, normalize: bool = True
    ) -> torch.Tensor:  # obtain the input image
        images = self.sensor.data.output[self.sensor_type]
        torch.nan_to_num_(images, nan=1e6)
        if normalize:
            images = self.norm_fn(images)
            images = images.permute(0, 3, 1, 2).contiguous()
        return images

    def _rgb_norm(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float() / 255.0
        mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
        images -= mean_tensor
        return images

    def _depth_norm(self, images: torch.Tensor) -> torch.Tensor:
        images = torch.tanh(images / 2) * 2
        images -= torch.mean(images, dim=(1, 2), keepdim=True)
        return images

    def show_collage(self, images: torch.Tensor, save_path: str = "collage.png"):
        import numpy as np
        from matplotlib import cm

        from PIL import Image

        a = images.detach().cpu().numpy()
        n, h, w, c = a.shape
        s = int(np.ceil(np.sqrt(n)))
        canvas = np.full((s * h, s * w, 3), 255, np.uint8)
        turbo = cm.get_cmap("turbo")
        for i in range(n):
            r, col = divmod(i, s)
            img = a[i]
            if c == 1:
                d = img[..., 0]
                d = (d - d.min()) / (np.ptp(d) + 1e-8)
                rgb = (turbo(d)[..., :3] * 255).astype(np.uint8)
            else:
                x = img if img.max() > 1 else img * 255
                rgb = np.clip(x, 0, 255).astype(np.uint8)
            canvas[r * h : (r + 1) * h, col * w : (col + 1) * w] = rgb
        Image.fromarray(canvas).save(save_path)


def time_left(env: ManagerBasedRLEnv):
    time_left_frac = 1 - env.episode_length_buf / env.max_episode_length
    return time_left_frac.view(env.num_envs, -1)
