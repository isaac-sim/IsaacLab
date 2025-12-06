# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import numpy as np
import torch

import omni.usd
from pxr import Gf, Usd, UsdGeom

from isaaclab.cloner import Cloner
from isaaclab.cloner.utils import replicate_environment
from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.utils.timer import Timer


class GridCloner(Cloner):
    """This is a specialized Cloner class that will automatically generate clones in a grid fashion."""

    def __init__(self, spacing: float, num_per_row: int = -1, stage: Usd.Stage = None):
        """
        Args:
            spacing (float): Spacing between clones.
            num_per_row (int): Number of clones to place in a row. Defaults to sqrt(num_clones).
            stage (Usd.Stage): Usd stage where source prim and clones are added to.
        """
        self._spacing = spacing
        self._num_per_row = num_per_row

        self._positions = None
        self._orientations = None

        Cloner.__init__(self, stage)

    def get_clone_transforms(
        self,
        num_clones: int,
        position_offsets: np.ndarray = None,
        orientation_offsets: np.ndarray = None,
    ):
        """Computes the positions and orientations of clones in a grid.

        Args:
            num_clones (int): Number of clones.
            position_offsets (np.ndarray): Positions to be applied as local translations on top of computed clone position.
            position_offsets (np.ndarray | torch.Tensor): Positions to be applied as local translations on top of computed clone position.
                                           Defaults to None, no offset will be applied.
            orientation_offsets (np.ndarray | torch.Tensor): Orientations to be applied as local rotations for each clone.
                                           Defaults to None, no offset will be applied.
        Returns:
            positions (List): Computed positions of all clones.
            orientations (List): Computed orientations of all clones.
        """
        # check if inputs are valid
        if position_offsets is not None:
            if len(position_offsets) != num_clones:
                raise ValueError("Dimension mismatch between position_offsets and prim_paths!")
            # convert to numpy array
            if isinstance(position_offsets, torch.Tensor):
                position_offsets = position_offsets.detach().cpu().numpy()
            elif not isinstance(position_offsets, np.ndarray):
                position_offsets = np.asarray(position_offsets)
        if orientation_offsets is not None:
            if len(orientation_offsets) != num_clones:
                raise ValueError("Dimension mismatch between orientation_offsets and prim_paths!")
            # convert to numpy array
            if isinstance(orientation_offsets, torch.Tensor):
                orientation_offsets = orientation_offsets.detach().cpu().numpy()
            elif not isinstance(orientation_offsets, np.ndarray):
                orientation_offsets = np.asarray(orientation_offsets)

        if self._positions is not None and self._orientations is not None:
            return self._positions, self._orientations

        self._num_per_row = int(np.sqrt(num_clones)) if self._num_per_row == -1 else self._num_per_row
        num_rows = np.ceil(num_clones / self._num_per_row)
        num_cols = np.ceil(num_clones / num_rows)

        row_offset = 0.5 * self._spacing * (num_rows - 1)
        col_offset = 0.5 * self._spacing * (num_cols - 1)

        positions = []
        orientations = []

        for i in range(num_clones):
            # compute transform
            row = i // num_cols
            col = i % num_cols
            y = row_offset - row * self._spacing
            x = col * self._spacing - col_offset

            up_axis = UsdGeom.GetStageUpAxis(self._stage)
            position = [y, x, 0] if up_axis == UsdGeom.Tokens.z else [x, 0, y]
            orientation = Gf.Quatd.GetIdentity()

            if position_offsets is not None:
                translation = position_offsets[i] + position
            else:
                translation = position

            if orientation_offsets is not None:
                orientation = (
                    Gf.Quatd(orientation_offsets[i][0].item(), Gf.Vec3d(orientation_offsets[i][1:].tolist()))
                    * orientation
                )

            orientation = [
                orientation.GetReal(),
                orientation.GetImaginary()[0],
                orientation.GetImaginary()[1],
                orientation.GetImaginary()[2],
            ]

            positions.append(translation)
            orientations.append(orientation)

        self._positions = positions
        self._orientations = orientations

        return positions, orientations

    @Timer(name="newton_clone", msg="Clone took:", enable=True, format="ms")
    def clone(
        self,
        source_prim_path: str,
        prim_paths: list[str],
        position_offsets: np.ndarray = None,
        orientation_offsets: np.ndarray = None,
        replicate_physics: bool = False,
        base_env_path: str = None,
        clone_in_fabric: bool = False,
        root_path: str = None,
        copy_from_source: bool = False,
        enable_env_ids: bool = False,
        spawn_offset: tuple[float] = (0.0, 0.0, 0.0),
    ):
        """Creates clones in a grid fashion. Positions of clones are computed automatically.

        Args:
            source_prim_path (str): Path of source object.
            prim_paths (List[str]): List of destination paths.
            position_offsets (np.ndarray): Positions to be applied as local translations on top of computed clone position.
                                           Defaults to None, no offset will be applied.
            orientation_offsets (np.ndarray): Orientations to be applied as local rotations for each clone.
                                           Defaults to None, no offset will be applied.
            replicate_physics (bool): Uses omni.physics replication. This will replicate physics properties directly for paths beginning with root_path and skip physics parsing for anything under the base_env_path.
            base_env_path (str): Path to namespace for all environments. Required if replicate_physics=True and define_base_env() not called.
            clone_in_fabric (bool): Not supported in Newton. This is here for compatibility with IL 2.2.
            root_path (str): Prefix path for each environment. Required if replicate_physics=True and generate_paths() not called.
            copy_from_source: (bool): Setting this to False will inherit all clones from the source prim; any changes made to the source prim will be reflected in the clones.
                         Setting this to True will make copies of the source prim when creating new clones; changes to the source prim will not be reflected in clones. Defaults to False. Note that setting this to True will take longer to execute.
            enable_env_ids (bool): Setting this enables co-location of clones in physics with automatic filtering of collisions between clones.
        Returns:
            positions (List): Computed positions of all clones.
        """

        num_clones = len(prim_paths)
        NewtonManager._num_envs = num_clones

        positions, orientations = self.get_clone_transforms(num_clones, position_offsets, orientation_offsets)
        if replicate_physics:
            clone_base_path = self._root_path if root_path is None else root_path
            builder, stage_info = replicate_environment(
                omni.usd.get_context().get_stage(),
                source_prim_path,
                clone_base_path + "{}",
                positions,
                orientations,
                spawn_offset=spawn_offset,
                simplify_meshes=True,
                collapse_fixed_joints=False,
                joint_ordering="dfs",
                joint_drive_gains_scaling=1.0,
            )
            NewtonManager.set_builder(builder)
        if not NewtonManager._clone_physics_only:
            super().clone(
                source_prim_path=source_prim_path,
                prim_paths=prim_paths,
                positions=positions,
                orientations=orientations,
                replicate_physics=replicate_physics,
                base_env_path=base_env_path,
                root_path=root_path,
                copy_from_source=copy_from_source,
                enable_env_ids=enable_env_ids,
            )
        return positions
