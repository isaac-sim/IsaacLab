# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import warp as wp
from newton import JointType
from newton import Model as NewtonModel
from newton.selection import ArticulationView

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
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


@dataclass
class _PoseDriver:
    """Per-pose-objective binding to the live articulation and the action vector.

    Holds the Isaac Lab body index used to read the current end-effector pose, the
    target-frame offset (batched to ``num_envs``), this objective's slice of the
    action vector, and the built solver objective. The per-step root-frame target
    buffers are allocated in :meth:`__post_init__` from the offset's shape/device.
    """

    body_idx: int
    offset_pos: torch.Tensor
    offset_rot: torch.Tensor
    slice: slice
    objective: NewtonIKPoseObjective
    target_pos_b: torch.Tensor = field(init=False)
    target_quat_b: torch.Tensor = field(init=False)

    def __post_init__(self):
        num_envs, device = self.offset_pos.shape[0], self.offset_pos.device
        self.target_pos_b = torch.zeros(num_envs, 3, device=device)
        self.target_quat_b = torch.zeros(num_envs, 4, device=device)
        self.target_quat_b[:, 3] = 1.0


class NewtonInverseKinematicsAction(ActionTerm):
    """Newton inverse-kinematics action term.

    The action solves IK as a single list of objectives on the single-env Newton
    prototype model registered by the cloner, then maps the resulting actuated
    joint coordinates back to the live batched Isaac Lab articulation. Each pose
    objective contributes its slice of the action vector and drives one
    end-effector body; a single pose objective is single-body IK, several are
    multi-body IK. Constraint objectives (e.g. joint limits) add no action
    dimensions. Fixed-base articulations only.
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
        # The prototype is the cloner's env_0 source builder; its single articulation
        # is addressed by the resolved source path.
        prototype_view = ArticulationView(
            prototype_model,
            self._source_path,
            verbose=False,
            exclude_joint_types=[JointType.FREE, JointType.FIXED],
        )
        self._prototype_joint_coord_ids = self._resolve_prototype_joint_coord_ids(
            prototype_view, self._asset.joint_names
        )
        self._prototype_controlled_coord_ids = self._resolve_prototype_joint_coord_ids(
            prototype_view, self._joint_names
        )
        self._prototype_joint_seed = wp.to_torch(prototype_model.joint_q).to(device=self.device, dtype=torch.float32)
        self._prototype_joint_seed = self._prototype_joint_seed.unsqueeze(0).repeat(self.num_envs, 1).contiguous()

        # The solver resolves each pose objective's body via the prototype view.
        self._ik_solver = self.cfg.controller.class_type(
            self.cfg.controller,
            model=prototype_model,
            num_envs=self.num_envs,
            device=self.device,
            objectives=self.cfg.objectives,
            link_resolver=lambda body_name: self._resolve_prototype_link_index(prototype_view, body_name),
        )

        # Bind each pose objective to the live articulation and the action vector.
        self._drivers: list[_PoseDriver] = []
        offset = 0
        for pose_cfg in pose_cfgs:
            name = pose_cfg.name if pose_cfg.name is not None else pose_cfg.body_name
            objective = self._ik_solver.objectives_by_name[name]
            body_idx = self._resolve_isaac_body_index(pose_cfg.body_name)
            offset_pos = torch.tensor(pose_cfg.body_offset_pos, device=self.device).repeat(self.num_envs, 1)
            offset_rot = torch.tensor(pose_cfg.body_offset_rot, device=self.device).repeat(self.num_envs, 1)
            self._drivers.append(
                _PoseDriver(body_idx, offset_pos, offset_rot, slice(offset, offset + objective.action_dim), objective)
            )
            offset += objective.action_dim
        self._action_dim = offset

        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

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
        # Each pose objective maps its own action slice (scaled internally) onto a
        # root-frame target from the body's current pose.
        for driver in self._drivers:
            ee_pos_b, ee_quat_b = self._compute_frame_pose(driver)
            target_pos_b, target_quat_b = driver.objective.compute_target_b(
                self._processed_actions[:, driver.slice], ee_pos_b, ee_quat_b
            )
            driver.target_pos_b[:] = target_pos_b
            driver.target_quat_b[:] = target_quat_b

    def apply_actions(self) -> None:
        # The IK solve runs on the single-env prototype model, so all batched
        # root-frame targets are expressed in the prototype (env 0) world frame.
        self._validate_matching_root_orientations()
        root_pos_proto = self._asset.data.root_pos_w.torch[0:1].repeat(self.num_envs, 1)
        root_quat_proto = self._asset.data.root_quat_w.torch[0:1].repeat(self.num_envs, 1)
        for driver in self._drivers:
            target_pos_w, target_quat_w = math_utils.combine_frame_transforms(
                root_pos_proto, root_quat_proto, driver.target_pos_b, driver.target_quat_b
            )
            driver.objective.set_target_pose(target_pos_w, target_quat_w)

        joint_seed = self._prototype_joint_seed.clone()
        joint_seed[:, self._prototype_joint_coord_ids] = self._asset.data.joint_pos.torch
        joint_pos_des_all = wp.to_torch(self._ik_solver.solve(wp.from_torch(joint_seed.contiguous(), dtype=wp.float32)))
        joint_pos_des = joint_pos_des_all[:, self._prototype_controlled_coord_ids].contiguous()
        self._asset.set_joint_position_target_index(target=joint_pos_des, joint_ids=self._joint_ids_warp)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[env_ids] = 0.0

    def _compute_frame_pose(self, driver: _PoseDriver) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._asset.data.body_pos_w.torch[:, driver.body_idx]
        ee_quat_w = self._asset.data.body_quat_w.torch[:, driver.body_idx]
        root_pos_w = self._asset.data.root_pos_w.torch
        root_quat_w = self._asset.data.root_quat_w.torch
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        ee_pos_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pos_b, ee_quat_b, driver.offset_pos, driver.offset_rot
        )
        return ee_pos_b, ee_quat_b

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
