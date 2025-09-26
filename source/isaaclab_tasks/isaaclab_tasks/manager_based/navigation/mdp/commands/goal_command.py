# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Code adapted from https://github.com/leggedrobotics/nav-suite

# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import CUBOID_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.sensors import RayCaster, RayCasterCamera
from isaaclab.utils.math import quat_apply_inverse, quat_from_angle_axis, wrap_to_pi, yaw_quat
from isaaclab.utils.warp import raycast_mesh

CYLINDER_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "cylinder": sim_utils.CylinderCfg(
            radius=1,
            height=1,
            axis="X",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    }
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .goal_command_cfg import GoalCommandCfg

# TODO: @pascal-roth: remove once multi-mesh raycasting is merged
try:
    from isaaclab.utils.warp import raycast_dynamic_meshes
except ImportError:
    raycast_dynamic_meshes = None


class GoalCommandTerm(CommandTerm):
    r"""Base class for goal commands.

    This class is used to define the common visualization features for goal commands.
    """

    cfg: GoalCommandCfg
    """Configuration for the command."""

    def __init__(self, cfg: GoalCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # -- generate height map
        self._construct_height_map()
        # -- construct traversability map
        self._construct_traversability_map()

        # -- goal commands: (x, y, z)
        self.pos_spawn_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_spawn_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)

    def __str__(self) -> str:
        msg = "GoalCommandGenerator:\n"
        msg += f"\tCommand dimension:\t {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range:\t {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base pose in base frame. Shape is (num_envs, 4)."""
        return torch.cat((self.pos_command_b, self.heading_command_b.unsqueeze(-1)), dim=1)

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the command for the specified environments."""

        # get the terrain id for the environments
        terrain_id = (
            torch.cdist(self._env.scene.terrain.env_origins[env_ids], self._terrain_origins, p=2)
            .argmin(dim=1)
            .unsqueeze(0)
        )

        start_sample = torch.concat(
            (terrain_id, torch.randint(0, self.split_max_length, (len(env_ids),), device=self.device).unsqueeze(0))
        )
        end_sample = torch.concat(
            (terrain_id, torch.randint(0, self.split_max_length, (len(env_ids),), device=self.device).unsqueeze(0))
        )

        # robot height
        robot_height = self.robot.data.default_root_state[env_ids, 2]

        # Update command buffers
        self.pos_command_w[env_ids] = self._split_traversability_map[start_sample[0, :], start_sample[1, :]]
        self.pos_command_w[env_ids, 2] = robot_height

        # Update spawn locations and heading buffer
        self.pos_spawn_w[env_ids] = self._split_traversability_map[end_sample[0, :], end_sample[1, :]]
        self.pos_spawn_w[env_ids, 2] = robot_height

        # Calculate the spawn heading based on the goal position
        self.heading_spawn_w[env_ids] = torch.atan2(
            self.pos_command_w[env_ids, 1] - self.pos_spawn_w[env_ids, 1],
            self.pos_command_w[env_ids, 0] - self.pos_spawn_w[env_ids, 0],
        )
        # Calculate the goal heading based on the goal position
        self.heading_command_w[env_ids] = torch.atan2(
            self.pos_command_w[env_ids, 1] - self.pos_spawn_w[env_ids, 1],
            self.pos_command_w[env_ids, 0] - self.pos_spawn_w[env_ids, 0],
        )

        # NOTE: the reset event is called before the new goal commands are generated, i.e. the spawn locations are
        # updated before the new goal commands are generated. To repsawn with the correct locations, we call here the
        # update spawn locations function
        if self.cfg.reset_pos_term_name:
            reset_term_idx = self._env.event_manager.active_terms["reset"].index(self.cfg.reset_pos_term_name)
            self._env.event_manager._mode_term_cfgs["reset"][reset_term_idx].func(
                self._env, env_ids, **self._env.event_manager._mode_term_cfgs["reset"][reset_term_idx].params
            )

    def _update_command(self):
        """Re-target the position command to the current root position and heading."""
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        target_vec[:, 2] = 0.0  # ignore z component
        self.pos_command_b[:] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)

        # update the heading command in the base frame
        # heading_w is angle world x axis to robot base x axis
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)

    def _update_metrics(self):
        """Update metrics."""
        self.metrics["error_pos"] = torch.norm(self.pos_command_w - self.robot.data.root_pos_w[:, :3], dim=1)

    """
    Helper functions
    """

    def _get_mesh_dimensions(
        self, raycaster: RayCaster | RayCasterCamera
    ) -> tuple[float, float, float, float, float, float]:
        # get min, max of the mesh in the xy plane
        # Get bounds of the terrain
        bounds = []

        def deep_flatten(nested_list):
            for item in nested_list:
                if isinstance(item, list):
                    yield from deep_flatten(item)  # recurse into sublist
                else:
                    yield item

        for mesh in list(deep_flatten(raycaster.meshes.values())):
            curr_bounds = torch.zeros((2, 3))
            curr_bounds[0] = torch.tensor(mesh.points).max(dim=0)[0]
            curr_bounds[1] = torch.tensor(mesh.points).min(dim=0)[0]
            bounds.append(curr_bounds)
        bounds = torch.vstack(bounds)
        x_min, y_min, z_min = bounds[:, 0].min().item(), bounds[:, 1].min().item(), bounds[:, 2].min().item()
        x_max, y_max, z_max = bounds[:, 0].max().item(), bounds[:, 1].max().item(), bounds[:, 2].max().item()
        return x_max, y_max, x_min, y_min, z_min, z_max

    def _construct_height_map(self):
        # get dimensions and construct height grid with raycasting
        raycaster: RayCaster | RayCasterCamera = self._env.scene.sensors[self.cfg.raycaster_sensor]

        # get mesh dimensions [x_max, y_max, x_min, y_min]
        mesh_dimensions = self._get_mesh_dimensions(raycaster)

        grid_x, grid_y = torch.meshgrid(
            torch.linspace(
                mesh_dimensions[2],
                mesh_dimensions[0],
                int(np.abs(np.ceil((mesh_dimensions[0] - mesh_dimensions[2]) / self.cfg.grid_resolution))),
                device=self.device,
            ),
            torch.linspace(
                mesh_dimensions[3],
                mesh_dimensions[1],
                int(np.abs(np.ceil((mesh_dimensions[1] - mesh_dimensions[3]) / self.cfg.grid_resolution))),
                device=self.device,
            ),
            indexing="ij",
        )
        grid_z = torch.ones_like(grid_x) * raycaster.cfg.max_distance
        self._height_grid_pos = torch.vstack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T
        direction = torch.zeros_like(self._height_grid_pos)
        direction[:, 2] = -1.0

        # check for collision with raycasting from the top
        # support for both multi-mesh and single-mesh raycasting
        if hasattr(raycaster, "_mesh_ids_wp") and raycast_dynamic_meshes is not None:
            hit_point = raycast_dynamic_meshes(
                ray_starts=self._height_grid_pos.unsqueeze(0),
                ray_directions=direction.unsqueeze(0),
                max_dist=raycaster.cfg.max_distance + 1e2,
                mesh_ids_wp=raycaster._mesh_ids_wp,
            )[0].squeeze(0)
        else:
            hit_point = raycast_mesh(
                ray_starts=self._height_grid_pos.unsqueeze(0),
                ray_directions=direction.unsqueeze(0),
                max_dist=raycaster.cfg.max_distance + 1e2,
                mesh=raycaster.meshes[raycaster.cfg.mesh_prim_paths[0]],
            )[0].squeeze(0)

        # get the height grid
        self._height_grid = hit_point[:, 2].reshape(
            int(np.abs(np.ceil((mesh_dimensions[0] - mesh_dimensions[2]) / self.cfg.grid_resolution))),
            int(np.abs(np.ceil((mesh_dimensions[1] - mesh_dimensions[3]) / self.cfg.grid_resolution))),
        )

    def _construct_traversability_map(self):
        # Define Sobel filters for x and y directions
        sobel_x = (
            torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        sobel_y = (
            torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # Apply the Sobel filters to the heigt_field_matrix while retraining the same shape
        edges_x = torch.nn.functional.conv2d(self._height_grid.unsqueeze(0).float(), sobel_x, padding=1)
        edges_y = torch.nn.functional.conv2d(self._height_grid.unsqueeze(0).float(), sobel_y, padding=1)

        # Compute the gradient magnitude (edge strength)
        edges = torch.sqrt(edges_x**2 + edges_y**2)

        edges_mask = edges > self.cfg.edge_threshold

        # Dilate the mask to expand the objects
        padding_size = int(self.cfg.robot_length / 2 / self.cfg.grid_resolution)
        kernel = torch.ones((1, 1, 2 * padding_size + 1, 2 * padding_size + 1), device=self.device)
        traversability_map = (
            torch.nn.functional.conv2d(edges_mask.float(), kernel, padding=padding_size).squeeze(1) < 0.5
        )
        traversability_map = traversability_map.reshape(-1, 1)

        # split the grid into subgrids for each terrain origin
        self._terrain_origins = self._env.scene.terrain.terrain_origins.reshape(-1, 3)
        point_distances = torch.cdist(self._height_grid_pos[..., :2], self._terrain_origins[..., :2], p=2)
        subgrid_ids = torch.argmin(point_distances, dim=1)

        # split the traversability map into the subgrids based on the subgrid_ids
        split_traversability_map = [
            self._height_grid_pos[subgrid_ids == i][traversability_map[subgrid_ids == i].squeeze()]
            for i in range(self._terrain_origins.shape[0])
        ]

        # make every snipped the same length by repeating prev. indexes
        split_lengths = [len(subgrid) for subgrid in split_traversability_map]
        self.split_max_length = int(np.median(split_lengths))

        self._split_traversability_map = torch.concat(
            [
                subgrid[torch.randint(0, max(split_lengths[idx], 1), (self.split_max_length,))].unsqueeze(0)
                for idx, subgrid in enumerate(split_traversability_map)
            ],
            dim=0,
        )

        # for cells where there is no accessible points, randomly assign points of another cell
        zero_split_lengths = np.array(split_lengths) == 0
        if np.sum(zero_split_lengths) > 0:
            while True:
                rand_cell_idx = torch.randint(0, len(split_traversability_map), (np.sum(zero_split_lengths),))
                if np.all(np.array(split_lengths)[rand_cell_idx] > 0):
                    break
            self._split_traversability_map[zero_split_lengths] = self._split_traversability_map[rand_cell_idx]


    """
    Visualization
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set the debug visualization for the command.

        Args:
            debug_vis (bool): Whether to enable debug visualization.
        """
        # create markers if necessary for the first time
        # for each marker type check that the correct command properties exist eg. need spawn position for spawn marker
        if debug_vis:
            if not hasattr(self, "box_goal_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/position_goal"
                marker_cfg.markers["cuboid"].size = (0.3, 0.3, 0.3)
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (1.0, 0.15, 0.0)
                marker_cfg.markers["cuboid"].visual_material.roughness = 0.7
                marker_cfg.markers["cuboid"].visual_material.metallic = 1.0
                self.box_goal_visualizer = VisualizationMarkers(marker_cfg)
                self.box_goal_visualizer.set_visibility(True)
            if self.cfg.vis_line and not hasattr(self, "line_to_goal_visualiser"):
                marker_cfg = CYLINDER_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/line_to_goal"
                marker_cfg.markers["cylinder"].height = 1
                marker_cfg.markers["cylinder"].radius = 0.05
                self.line_to_goal_visualiser = VisualizationMarkers(marker_cfg)
                self.line_to_goal_visualiser.set_visibility(True)
        else:
            if hasattr(self, "box_goal_visualizer"):
                self.box_goal_visualizer.set_visibility(False)
            if self.cfg.vis_line and hasattr(self, "line_to_goal_visualiser"):
                self.line_to_goal_visualiser.set_visibility(False)

    def _debug_vis_callback(self, event, env_ids: Sequence[int] | None = None):
        """Callback function for the debug visualization."""
        if env_ids is None:
            env_ids = slice(None)

        # update goal marker if it exists
        self.box_goal_visualizer.visualize(self.pos_command_w[env_ids])

        if self.cfg.vis_line:
            # update the line marker
            # calculate the difference vector between the robot root position and the goal position
            # TODO @tasdep this assumes that robot.data.body_pos_w exists
            difference = self.pos_command_w - self.robot.data.body_pos_w[:, 0, :3]
            translations = self.robot.data.body_pos_w[:, 0, :3]
            # calculate the scale of the arrow (Mx3)
            difference_norm = torch.norm(difference, dim=1)
            # translate half of the length along difference axis
            translations += difference / 2
            # scale along x axis
            scales = torch.vstack(
                [difference_norm, torch.ones_like(difference_norm), torch.ones_like(difference_norm)]
            ).T
            # convert the difference vector to a quaternion
            difference = torch.nn.functional.normalize(difference, dim=1)
            x_vec = torch.tensor([1, 0, 0]).float().to(self.pos_command_w.device)
            angle = -torch.acos(difference @ x_vec)
            axis = torch.linalg.cross(difference, x_vec.expand_as(difference))
            quat = quat_from_angle_axis(angle, axis)

            # apply transforms
            self.line_to_goal_visualiser.visualize(
                translations=translations[env_ids], scales=scales[env_ids], orientations=quat[env_ids]
            )
