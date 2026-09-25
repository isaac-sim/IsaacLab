# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Four-slot policy playback over a physical parcel pool and USD-authored warehouse."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.common import VecEnvStepReturn

from .conveyor_cube_pool import ConveyorCubePool
from .conveyor_franka_env import ConveyorFrankaEnv

if TYPE_CHECKING:
    from pxr import Usd

    from .conveyor_franka_asset_env_cfg import ConveyorFrankaA09A12EnvCfg


class ConveyorFrankaWarehouseEnv(ConveyorFrankaEnv):
    """Play the four-cube checkpoint over an individually tracked workcell parcel pool.

    Kit renders the authored USD materials, lights, and animation. Lightweight
    Newton viewers show a static approximation of the warehouse dressing.
    """

    def __init__(self, cfg: ConveyorFrankaA09A12EnvCfg, render_mode: str | None = None, **kwargs):
        self._warehouse_animation: list[tuple[Usd.Attribute, Usd.Attribute]] = []
        self._warehouse_arm_joint_ids: list[int] | None = None
        cfg.scene._configure_route_assets(cfg.commands.transfer.parcel_colors)
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        if not any(viz.cfg.visualizer_type == "kit" for viz in self.sim.visualizers):
            return
        self.sim.set_setting("/app/viewport/grid/enabled", False)

        from pxr import Usd

        from .conveyor_franka_asset_env_cfg import _presentation_layer

        camera = self.sim.stage.GetPrimAtPath("/OmniverseKit_Persp")
        if camera:
            # Kit owns this camera in its session layer; weaker root-layer edits are ignored.
            viewer = cfg.sim.default_visualizer_cfg
            with Usd.EditContext(self.sim.stage, self.sim.stage.GetSessionLayer()):
                self.sim.set_camera_view(viewer.eye, viewer.lookat)
                camera.GetAttribute("focalLength").Set(viewer.focal_length)
        self._warehouse_source = Usd.Stage.Open(_presentation_layer(cfg.scene.warehouse_visual.spawn.usd_path))
        self._warehouse_period = self._warehouse_source.GetEndTimeCode()
        self._warehouse_fps = self._warehouse_source.GetTimeCodesPerSecond()
        for env_path in self.scene.env_prim_paths:
            group = self._warehouse_source.GetPrimAtPath("/Warehouse/Parcels")
            for source in group.GetChildren():
                target = self.sim.stage.GetPrimAtPath(f"{env_path}/WarehouseVisual/Parcels/{source.GetName()}")
                for name in ("xformOp:translate", "xformOp:rotateXYZ"):
                    self._warehouse_animation.append((source.GetAttribute(name), target.GetAttribute(name)))
        self.sim.add_render_callback("conveyor_warehouse_animation", self._animate_warehouse)

    def load_managers(self) -> None:
        """Create slot identities before command, reward, and observation terms inspect cubes."""
        assets = tuple(self.scene[f"cube_{i}"] for i in range(self.cfg.conveyor_force.transported_body_count_per_env))
        self.conveyor_cube_pool = ConveyorCubePool(assets, self.num_envs, self.device)
        super().load_managers()

    @staticmethod
    def _in_workcell(positions: torch.Tensor) -> torch.Tensor:
        """Identify parcels within the trained controller's local manipulation region [m]."""
        return (
            (positions[..., 0] > -0.25)
            & (positions[..., 0] < 1.5)
            & (positions[..., 1].abs() < 0.6)
            & (positions[..., 2] < 0.4)
            # The low merge passes beside the placement bend but is still remote transport.
            & ((positions[..., 2] < 0.12) | (positions[..., 0] < 1.05))
        )

    def _adapt_policy_cube_state(self, positions, quaternions, velocities):
        """Represent remote inventory as waiting slots while keeping local manipulation states exact."""
        origins = self.scene.env_origins[:, None, :]
        local = positions - origins
        remote = ~self._in_workcell(local)
        waiting = local.clone()
        waiting[..., 0] = 0.14 + 0.88 * torch.arange(1, 5, device=self.device)[None, :] / 5
        waiting[..., 1] = torch.where(local[..., 1] >= 0, 0.75, -0.75)
        waiting[..., 2] = 0.06
        upright = torch.zeros_like(quaternions)
        upright[..., 3] = 1.0
        transport_velocity = torch.zeros_like(velocities)
        transport_velocity[..., 0] = self.cfg.conveyor_force.speed
        return (
            torch.where(remote[..., None], waiting + origins, positions),
            torch.where(remote[..., None], upright, quaternions),
            torch.where(remote[..., None], transport_velocity, velocities),
        )

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Park while waiting for a misplaced parcel; dispatch runs through the command manager."""
        command = self.command_manager.get_term("transfer")
        robot = self.scene["robot"]
        if self._warehouse_arm_joint_ids is None:
            self._warehouse_arm_joint_ids = robot.find_joints(
                self.cfg.actions.arm_action.joint_names, preserve_order=True
            )[0]
        joint_ids = self._warehouse_arm_joint_ids
        parked = torch.zeros_like(action)
        parked[:, :7] = (
            (robot.data.default_joint_pos.torch[:, joint_ids] - robot.data.joint_pos.torch[:, joint_ids])
            / self.cfg.actions.arm_action.scale
        ).clamp(-0.25, 0.25)
        # Keep invalid actions visible to the existing sanitization and termination terms.
        use_policy = command.has_target | ~torch.isfinite(action).all(dim=1)
        return super().step(torch.where(use_policy[:, None], action, parked))

    def _animate_warehouse(self, _event) -> None:
        """Sample USD motion using policy time, independently of render frame rate."""
        from pxr import Sdf

        time_code = (self.common_step_counter * self.step_dt * self._warehouse_fps) % self._warehouse_period
        with Sdf.ChangeBlock():
            for source, target in self._warehouse_animation:
                target.Set(source.Get(time_code))

    def _reset_idx(self, env_ids) -> None:
        """Shuffle a mixed batch across the authored feeds in selected environments."""
        from .conveyor_warehouse_geometry import warehouse_parcel_positions

        pool = self.conveyor_cube_pool
        pool.reset(env_ids)
        super()._reset_idx(env_ids)
        positions = torch.tensor(warehouse_parcel_positions(), device=self.device)
        if self.cfg.commands.transfer.randomize_arrivals:
            assignments = torch.rand((len(env_ids), len(pool.assets)), device=self.device).argsort(dim=1)
        else:
            assignments = torch.arange(len(pool.assets), device=self.device).expand(len(env_ids), -1)
        for cube_id, cube in enumerate(pool.assets):
            pose = cube.data.root_pose_w.torch[env_ids].clone()
            pose[:, :3] = positions[assignments[:, cube_id]] + self.scene.env_origins[env_ids]
            pose[:, 3:] = pose.new_tensor((0.0, 0.0, 0.0, 1.0))
            cube.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
            cube.write_root_velocity_to_sim_index(root_velocity=pose.new_zeros((len(env_ids), 6)), env_ids=env_ids)

    def close(self) -> None:
        """Remove the presentation callback before releasing the shared task resources."""
        if getattr(self, "sim", None) is not None:
            self.sim.remove_render_callback("conveyor_warehouse_animation")
        self._warehouse_animation.clear()
        super().close()
