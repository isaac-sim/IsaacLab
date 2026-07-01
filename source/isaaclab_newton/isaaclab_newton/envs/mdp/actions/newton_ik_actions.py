# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import warp as wp
from newton import JointType
from newton import Model as NewtonModel
from newton.selection import ArticulationView

import isaaclab.sim as sim_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation.base_articulation import BaseArticulation
from isaaclab.cloner import resolve_clone_plan_source
from isaaclab.managers.action_manager import ActionTerm

from isaaclab_newton.ik.newton_ik_objectives_cfg import NewtonIKPoseObjectiveCfg
from isaaclab_newton.physics import NewtonManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from isaaclab_newton.ik.newton_ik_objectives import NewtonIKPoseObjective

    from .newton_ik_actions_cfg import NewtonInverseKinematicsActionCfg


logger = logging.getLogger(__name__)


@wp.kernel(enable_backward=False)
def _ik_world_target_kernel(
    body_pos_w: wp.array2d(dtype=wp.vec3f),
    body_quat_w: wp.array2d(dtype=wp.quatf),
    root_pos_w: wp.array(dtype=wp.vec3f),
    root_quat_w: wp.array(dtype=wp.quatf),
    body_idx: int,
    offset: wp.transformf,
    action: wp.array2d(dtype=wp.float32),
    action_offset: int,
    scale: wp.array(dtype=wp.float32),
    command_code: int,
    use_relative: int,
    out_pos: wp.array(dtype=wp.vec3f),
    out_rot: wp.array(dtype=wp.vec4f),
):
    """Map one pose objective's action slice to a prototype-world target pose.

    Mirrors ``subtract_frame_transforms`` -> body offset -> command (position /
    relative-pose / absolute-pose) -> ``combine_frame_transforms`` against the
    env-0 root, writing the target straight into the objective's Warp arrays.
    """
    i = wp.tid()
    # End-effector (offset) pose in the env's root frame.
    root_t = wp.transformf(root_pos_w[i], root_quat_w[i])
    ee_t = wp.transform_multiply(
        wp.transform_inverse(root_t), wp.transformf(body_pos_w[i, body_idx], body_quat_w[i, body_idx])
    )
    ee_t = wp.transform_multiply(ee_t, offset)
    ee_pos = wp.transform_get_translation(ee_t)
    ee_rot = wp.transform_get_rotation(ee_t)

    target_pos = ee_pos
    target_rot = ee_rot
    if command_code == 0:  # COMMAND_POSITION
        disp = wp.vec3f(
            action[i, action_offset + 0] * scale[0],
            action[i, action_offset + 1] * scale[1],
            action[i, action_offset + 2] * scale[2],
        )
        target_pos = ee_pos + disp if use_relative == 1 else disp
    else:
        if use_relative == 1:
            target_pos = ee_pos + wp.vec3f(
                action[i, action_offset + 0] * scale[0],
                action[i, action_offset + 1] * scale[1],
                action[i, action_offset + 2] * scale[2],
            )
            rot_vec = wp.vec3f(
                action[i, action_offset + 3] * scale[3],
                action[i, action_offset + 4] * scale[4],
                action[i, action_offset + 5] * scale[5],
            )
            angle = wp.length(rot_vec)
            delta_rot = wp.quat_identity()
            if angle > 1.0e-6:
                delta_rot = wp.quat_from_axis_angle(rot_vec / angle, angle)
            target_rot = delta_rot * ee_rot
        else:
            target_pos = wp.vec3f(
                action[i, action_offset + 0] * scale[0],
                action[i, action_offset + 1] * scale[1],
                action[i, action_offset + 2] * scale[2],
            )
            target_rot = wp.quatf(
                action[i, action_offset + 3] * scale[3],
                action[i, action_offset + 4] * scale[4],
                action[i, action_offset + 5] * scale[5],
                action[i, action_offset + 6] * scale[6],
            )

    # Broadcast against the env-0 prototype root (all roots identical, validated).
    world_t = wp.transform_multiply(wp.transformf(root_pos_w[0], root_quat_w[0]), wp.transformf(target_pos, target_rot))
    out_pos[i] = wp.transform_get_translation(world_t)
    q = wp.transform_get_rotation(world_t)
    out_rot[i] = wp.vec4f(q[0], q[1], q[2], q[3])


@wp.kernel(enable_backward=False)
def _ik_seed_scatter_kernel(
    joint_pos: wp.array2d(dtype=wp.float32),
    coord_ids: wp.array(dtype=wp.int32),
    seed: wp.array2d(dtype=wp.float32),
):
    """Overwrite the seed's actuated coordinates with the live joint positions."""
    i, j = wp.tid()
    seed[i, coord_ids[j]] = joint_pos[i, j]


@wp.kernel(enable_backward=False)
def _ik_gather_kernel(
    solved: wp.array2d(dtype=wp.float32),
    coord_ids: wp.array(dtype=wp.int32),
    out: wp.array2d(dtype=wp.float32),
):
    """Gather the controlled coordinates from the full solved joint vector."""
    i, k = wp.tid()
    out[i, k] = solved[i, coord_ids[k]]


@dataclass
class _PoseDriver:
    """Per-pose-objective binding: the live body to read and its action slice offset."""

    body_idx: int
    action_offset: int
    objective: NewtonIKPoseObjective


class NewtonInverseKinematicsAction(ActionTerm):
    """Newton inverse-kinematics action term.

    Solves IK as a single list of objectives on the cloner's single-env Newton
    prototype model, then maps the actuated joint coordinates back to the live
    batched articulation. Each pose objective drives one end-effector body (one is
    single-body IK, several are multi-body); constraint objectives add no action
    dimensions. The per-step target computation, seed assembly, solve and gather
    run entirely in Warp -- Torch appears only as the policy action at the
    boundary, viewed zero-copy into Warp. Fixed-base articulations only.
    """

    cfg: NewtonInverseKinematicsActionCfg
    _asset: BaseArticulation

    def __init__(self, cfg: NewtonInverseKinematicsActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        if not self._asset.is_fixed_base:
            raise ValueError("NewtonInverseKinematicsAction currently supports fixed-base articulations only.")

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._joint_ids_warp = wp.array(self._joint_ids, dtype=wp.int32, device=self.device)

        pose_cfgs = [obj for obj in self.cfg.objectives if isinstance(obj, NewtonIKPoseObjectiveCfg)]
        if not pose_cfgs:
            raise ValueError("NewtonInverseKinematicsAction requires at least one pose objective.")

        # Resolve the controlled asset to its clone-plan source and finalize the
        # single-env prototype builder the cloner already retained -- the same
        # source resolution other Newton consumers use, no bespoke registry.
        plan = sim_utils.SimulationContext.instance().get_clone_plan()
        source_path, _, asset_suffix = resolve_clone_plan_source(self._asset.cfg.prim_path, plan)
        # The proto builder is keyed by the bare clone source; the articulation
        # lives at the asset suffix below it (e.g. ".../env_0" + "/Robot").
        self._source_path = source_path + asset_suffix
        prototype_model = NewtonManager._cl_protos[source_path].finalize(device=NewtonManager.get_model().device)
        prototype_view = ArticulationView(
            prototype_model,
            self._source_path,
            verbose=False,
            exclude_joint_types=[JointType.FREE, JointType.FIXED],
        )
        coord_ids = self._resolve_prototype_joint_coord_ids(prototype_view, self._asset.joint_names)
        controlled_ids = self._resolve_prototype_joint_coord_ids(prototype_view, self._joint_names)

        # The solver resolves each pose objective's body via the prototype view.
        self._ik_solver = self.cfg.controller.class_type(
            self.cfg.controller,
            model=prototype_model,
            num_envs=self.num_envs,
            device=self.device,
            objectives=self.cfg.objectives,
            link_resolver=lambda body_name: self._resolve_prototype_link_index(prototype_view, body_name),
        )

        # Bind each pose objective to the live body it reads and its action slice.
        self._drivers: list[_PoseDriver] = []
        offset = 0
        for pose_cfg in pose_cfgs:
            name = pose_cfg.name if pose_cfg.name is not None else pose_cfg.body_name
            objective = self._ik_solver.objectives_by_name[name]
            body_idx = self._resolve_isaac_body_index(pose_cfg.body_name)
            self._drivers.append(_PoseDriver(body_idx, offset, objective))
            offset += objective.action_dim
        self._action_dim = offset

        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # Warp scratch for the seed -> solve -> gather pipeline.
        num_coords = prototype_model.joint_coord_count
        default_seed = wp.to_torch(prototype_model.joint_q).to(device=self.device, dtype=torch.float32)
        self._default_seed = wp.from_torch(default_seed.unsqueeze(0).repeat(self.num_envs, 1).contiguous())
        self._seed = wp.zeros((self.num_envs, num_coords), dtype=wp.float32, device=self.device)
        self._joint_pos_des = wp.zeros((self.num_envs, len(self._joint_ids)), dtype=wp.float32, device=self.device)
        self._coord_ids = wp.from_torch(coord_ids.to(torch.int32).contiguous())
        self._controlled_ids = wp.from_torch(controlled_ids.to(torch.int32).contiguous())

        self._clip = None
        if self.cfg.clip is not None:
            self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                self.num_envs, self._action_dim, 1
            )
            action_names = self._action_coordinate_names()
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, action_names)
            self._clip[:, index_list] = torch.tensor(value_list, device=self.device)

        logger.info(
            "Resolved Newton IK action joints %s [%s] and bodies %s.",
            self._joint_names,
            self._joint_ids,
            [(d.objective.name, d.body_idx) for d in self._drivers],
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        super().IO_descriptor
        self._IO_descriptor.shape = (self._action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "NewtonInverseKinematicsAction"
        self._IO_descriptor.joint_names = self._joint_names
        self._IO_descriptor.clip = self.cfg.clip
        self._IO_descriptor.extras["controller_cfg"] = self.cfg.controller.__dict__
        self._IO_descriptor.extras["objective_names"] = [d.objective.name for d in self._drivers]
        self._IO_descriptor.extras["coordinate_names"] = self._action_coordinate_names()
        return self._IO_descriptor

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._raw_actions
        if self._clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        # Each pose objective maps its action slice to a prototype-world target,
        # written straight into its Warp target arrays.
        self._validate_matching_root_orientations()
        action_wp = wp.from_torch(self._processed_actions.contiguous(), dtype=wp.float32)
        body_pos_w = self._asset.data.body_pos_w.warp
        body_quat_w = self._asset.data.body_quat_w.warp
        root_pos_w = self._asset.data.root_pos_w.warp
        root_quat_w = self._asset.data.root_quat_w.warp
        for driver in self._drivers:
            obj = driver.objective
            wp.launch(
                _ik_world_target_kernel,
                dim=self.num_envs,
                inputs=[
                    body_pos_w,
                    body_quat_w,
                    root_pos_w,
                    root_quat_w,
                    driver.body_idx,
                    obj.offset,
                    action_wp,
                    driver.action_offset,
                    obj.scale,
                    obj.command_code,
                    obj.use_relative,
                    obj.position_objective.target_positions,
                    obj.rotation_objective.target_rotations,
                ],
                device=self.device,
            )

    def apply_actions(self) -> None:
        # Seed the solver from the live joint positions on top of the prototype
        # default, solve, and write the controlled coordinates back -- all in Warp.
        wp.copy(self._seed, self._default_seed)
        wp.launch(
            _ik_seed_scatter_kernel,
            dim=(self.num_envs, len(self._asset.joint_names)),
            inputs=[self._asset.data.joint_pos.warp, self._coord_ids, self._seed],
            device=self.device,
        )
        solved = self._ik_solver.solve(self._seed)
        wp.launch(
            _ik_gather_kernel,
            dim=(self.num_envs, len(self._joint_ids)),
            inputs=[solved, self._controlled_ids, self._joint_pos_des],
            device=self.device,
        )
        self._asset.set_joint_position_target_index(target=self._joint_pos_des, joint_ids=self._joint_ids_warp)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[env_ids] = 0.0

    def _validate_matching_root_orientations(self) -> None:
        """Guard the prototype-frame IK assumption for replicated fixed-base roots."""
        root_quat_w = self._asset.data.root_quat_w.torch
        if root_quat_w.shape[0] <= 1:
            return
        # q and -q represent the same orientation, so compare absolute dot products.
        same_orientation = torch.abs(torch.sum(root_quat_w * root_quat_w[0:1], dim=-1)) > 1.0 - 1e-5
        if not torch.all(same_orientation):
            bad_env_ids = torch.nonzero(~same_orientation, as_tuple=False).flatten().tolist()
            raise RuntimeError(
                "NewtonInverseKinematicsAction solves against the env 0 prototype root orientation, but "
                f"root orientations differ in env ids {bad_env_ids}. Use identical fixed-base root orientations "
                "for this action."
            )

    def _resolve_isaac_body_index(self, body_name: str) -> int:
        body_ids, body_names = self._asset.find_bodies(body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for Newton IK body_name={body_name}. Found {len(body_ids)}: {body_names}."
            )
        return body_ids[0]

    def _resolve_prototype_joint_coord_ids(
        self, prototype_view: ArticulationView, joint_names: Sequence[str]
    ) -> torch.Tensor:
        layout = prototype_view.frequency_layouts[NewtonModel.AttributeFrequency.JOINT_COORD]
        selected_indices = self._layout_indices(layout)
        coord_indices_by_name = {
            name: layout.offset + selected_indices[index] for index, name in enumerate(prototype_view.joint_coord_names)
        }
        coord_ids = [coord_indices_by_name[name] for name in joint_names]
        return torch.tensor(coord_ids, device=self.device, dtype=torch.long)

    def _resolve_prototype_link_index(self, prototype_view: ArticulationView, body_name: str) -> int:
        layout = prototype_view.frequency_layouts[NewtonModel.AttributeFrequency.BODY]
        selected_indices = self._layout_indices(layout)
        local_link_index = prototype_view.link_names.index(body_name)
        return layout.offset + selected_indices[local_link_index]

    @staticmethod
    def _layout_indices(layout) -> list[int]:
        if layout.slice is not None:
            return list(range(layout.slice.start, layout.slice.stop))
        return [int(index) for index in layout.indices.numpy().tolist()]

    def _action_coordinate_names(self) -> list[str]:
        names: list[str] = []
        for driver in self._drivers:
            names.extend(f"{driver.objective.name}/{coord}" for coord in driver.objective.command_coordinate_names())
        return names
