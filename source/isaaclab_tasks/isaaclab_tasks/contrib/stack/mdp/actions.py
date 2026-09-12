# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific actions for reset-table cube stacking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.joint_actions import JointAction

from .runtime_state import get_stack_reset_runtime_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import (
        ResetBufferedGripperActionCfg,
        ResetPreservingRelativeJointPositionActionCfg,
        WorkspaceBoundedRelativeJointPositionActionCfg,
    )


class ResetBufferedGripperAction(BinaryJointPositionAction):
    """Keep reset-supplied grasps closed only while the reset state settles.

    Newton Cube Lift keeps the physical gripper closed throughout the
    reset-assisted grasp portion of its curriculum. Stacking can use that
    protection only during the reset grace period: continuing to mask an open
    policy command through transport would credit an action that drops the
    cube in an uninterrupted table-start episode. Raw actions remain visible
    to the termination manager, and after the grace period the policy owns the
    physical gripper completely.
    """

    cfg: ResetBufferedGripperActionCfg

    def process_actions(self, actions: torch.Tensor) -> None:
        """Map binary policy actions and protect reset-assisted acquisition."""
        super().process_actions(actions)
        if self.cfg.force_close_steps == 0:
            return
        held_cube_ids = get_stack_reset_runtime_state(self._env).held_cube_ids
        force_close = (held_cube_ids >= 0) & (self._env.episode_length_buf < self.cfg.force_close_steps)
        self._processed_actions[force_close] = self._close_command


class ResetPreservingRelativeJointPositionAction(JointAction):
    """Apply bounded DexSuite-relative targets with a reset-grasp handoff.

    Every policy residual is added to the measured joint position exactly once
    per policy step, matching Isaac Lab's standard
    :class:`RelativeJointPositionAction`. This prevents exploration noise from
    random-walking an unobserved 16-dimensional target into the soft limits.

    Held reset rows optionally anchor the target at a pair-conditioned preload
    until the policy deliberately commands opening for a short debounce. Policy
    residuals remain active around the anchor, so the handoff does not mask PPO
    actions. Once released, the anchor cannot rearm before the next reset.
    """

    cfg: ResetPreservingRelativeJointPositionActionCfg

    def __init__(self, cfg: ResetPreservingRelativeJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        if cfg.max_delta <= 0.0:
            raise ValueError("max_delta must be positive.")
        if cfg.joint_limit_margin < 0.0:
            raise ValueError("joint_limit_margin must be non-negative.")
        if not 0.0 < cfg.preload_release_threshold <= 1.0:
            raise ValueError("preload_release_threshold must lie in (0, 1].")
        if cfg.preload_release_steps < 1:
            raise ValueError("preload_release_steps must be positive.")
        self._torch_joint_ids = self._joint_ids.long() if isinstance(self._joint_ids, torch.Tensor) else self._joint_ids
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._torch_joint_ids]
        if torch.any(limits[..., 0] + cfg.joint_limit_margin >= limits[..., 1] - cfg.joint_limit_margin):
            raise ValueError("joint_limit_margin leaves at least one controlled joint without a valid range.")
        self._position_targets = self._asset.data.joint_pos.torch[:, self._torch_joint_ids].clone()
        self._pair_reset_preload_commands: torch.Tensor | None = None
        self._pair_reset_open_commands: torch.Tensor | None = None
        self._preload_assist_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._preload_open_intent_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        command_joint_names = tuple(cfg.reset_preload_joint_names)
        preload_commands_by_pair = cfg.reset_preload_commands_by_pair
        open_commands_by_pair = cfg.reset_open_commands_by_pair
        if command_joint_names and not preload_commands_by_pair:
            raise ValueError("reset_preload_joint_names requires reset_preload_commands_by_pair.")
        if preload_commands_by_pair and not command_joint_names:
            raise ValueError("reset_preload_commands_by_pair requires reset_preload_joint_names.")
        if bool(preload_commands_by_pair) != bool(open_commands_by_pair):
            raise ValueError("Reset preload and open commands must be configured together.")
        if preload_commands_by_pair:
            if len(command_joint_names) != self._num_joints or set(command_joint_names) != set(self._joint_names):
                raise ValueError(
                    "reset_preload_joint_names must contain every controlled joint exactly once; "
                    f"resolved {tuple(self._joint_names)}, configured {command_joint_names}."
                )
            preload_commands = torch.tensor(preload_commands_by_pair, dtype=torch.float32, device=self.device)
            open_commands = torch.tensor(open_commands_by_pair, dtype=torch.float32, device=self.device)
            if preload_commands.ndim != 2 or preload_commands.shape[1] != self._num_joints:
                raise ValueError(
                    "reset_preload_commands_by_pair must have shape "
                    f"(num_pairs, {self._num_joints}); received {tuple(preload_commands.shape)}."
                )
            if open_commands.shape != preload_commands.shape:
                raise ValueError(
                    "reset_open_commands_by_pair must match reset_preload_commands_by_pair; "
                    f"received {tuple(open_commands.shape)} and {tuple(preload_commands.shape)}."
                )
            if preload_commands.shape[0] < 1:
                raise ValueError("At least one pair-conditioned reset preload command must be configured.")
            if not torch.all(torch.isfinite(preload_commands)) or not torch.all(torch.isfinite(open_commands)):
                raise ValueError("Reset preload and open commands must contain only finite values.")

            name_to_command_id = {name: command_id for command_id, name in enumerate(command_joint_names)}
            resolved_order = torch.tensor(
                [name_to_command_id[name] for name in self._joint_names],
                dtype=torch.long,
                device=self.device,
            )
            self._pair_reset_preload_commands = preload_commands[:, resolved_order]
            self._pair_reset_open_commands = open_commands[:, resolved_order]

    def process_actions(self, actions: torch.Tensor) -> None:
        """Build one measured-state target and apply reset-only preload assist."""
        super().process_actions(actions)
        target_delta = torch.clamp(
            self._processed_actions,
            min=-self.cfg.max_delta,
            max=self.cfg.max_delta,
        )
        current_position = self._asset.data.joint_pos.torch[:, self._torch_joint_ids]
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._torch_joint_ids]
        lower = limits[..., 0] + self.cfg.joint_limit_margin
        upper = limits[..., 1] - self.cfg.joint_limit_margin
        position_targets = current_position + target_delta

        if self._pair_reset_preload_commands is not None:
            grasp_pair_ids = get_stack_reset_runtime_state(self._env).grasp_pair_ids.long()
            safe_pair_ids = torch.clamp(
                grasp_pair_ids,
                min=0,
                max=self._pair_reset_preload_commands.shape[0] - 1,
            )
            invalid_pair_ids = self._preload_assist_active & (grasp_pair_ids != safe_pair_ids)
            if torch.any(invalid_pair_ids):
                invalid_values = torch.unique(grasp_pair_ids[invalid_pair_ids]).tolist()
                raise ValueError(
                    f"Held reset rows reference grasp-pair IDs without preload commands: {invalid_values}."
                )
            preload_targets = torch.clamp(
                self._pair_reset_preload_commands[safe_pair_ids],
                min=lower,
                max=upper,
            )
            open_targets = torch.clamp(
                self._pair_reset_open_commands[safe_pair_ids],
                min=lower,
                max=upper,
            )
            opening_direction = open_targets - preload_targets
            opening_fraction = torch.sum(target_delta * opening_direction, dim=1) / torch.sum(
                torch.square(opening_direction),
                dim=1,
            ).clamp_min(1.0e-12)
            opening_intent = self._preload_assist_active & (opening_fraction >= self.cfg.preload_release_threshold)
            self._preload_open_intent_steps = torch.where(
                opening_intent,
                self._preload_open_intent_steps + 1,
                torch.zeros_like(self._preload_open_intent_steps),
            )
            release = self._preload_assist_active & (self._preload_open_intent_steps >= self.cfg.preload_release_steps)
            self._preload_assist_active[release] = False
            self._preload_open_intent_steps[release] = 0
            position_targets = torch.where(
                self._preload_assist_active.unsqueeze(-1),
                preload_targets + target_delta,
                position_targets,
            )

        self._position_targets = torch.clamp(position_targets, min=lower, max=upper)
        self._processed_actions = self._position_targets
        # Preserve the normalized policy output for observations and
        # action-rate regularization.
        self._raw_actions[:] = actions

    def apply_actions(self) -> None:
        """Hold the policy-step position target through every physics substep."""
        self._asset.set_joint_position_target_index(target=self.processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Initialize targets from the sampled reset pose."""
        super().reset(env_ids)
        current_position = self._asset.data.joint_pos.torch[:, self._torch_joint_ids]
        if env_ids is None:
            self._position_targets[:] = current_position
            self._processed_actions[:] = current_position
        else:
            self._position_targets[env_ids] = current_position[env_ids]
            self._processed_actions[env_ids] = current_position[env_ids]
        held_cube_ids = get_stack_reset_runtime_state(self._env).held_cube_ids
        self._preload_assist_active[env_ids] = held_cube_ids[env_ids] >= 0
        self._preload_open_intent_steps[env_ids] = 0


class WorkspaceBoundedRelativeJointPositionAction(JointAction):
    """Map policy joint deltas to gravity-compensated, bounded position targets.

    Each policy output directly controls the corresponding arm joint. A scaled
    delta is added to the measured joint position exactly once
    per policy step, then the resulting target is held through every physics
    substep. This preserves standard relative-joint semantics without applying
    the same delta repeatedly during simulation decimation. Targets are also
    clamped to both the robot's soft limits and the reset-buffer workspace.
    Model-based gravity effort can be added alongside the impedance target.
    """

    cfg: WorkspaceBoundedRelativeJointPositionActionCfg

    @property
    def controller_owned_write_methods(self) -> tuple[str, ...]:
        """Exclude controller-owned gravity feedforward from policy outputs."""
        if self.cfg.controller_owns_gravity_compensation:
            return ("set_joint_effort_target_index",)
        return ()

    def __init__(self, cfg: WorkspaceBoundedRelativeJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        if cfg.controller_owns_gravity_compensation and not cfg.gravity_compensation:
            raise ValueError("Controller-owned gravity compensation requires gravity_compensation=True.")
        # Newton exposes resolved indices as int32 for Warp kernels, while PyTorch/ONNX
        # advanced indexing requires int64 indices. Keep both representations explicit.
        self._torch_joint_ids = self._joint_ids.long() if isinstance(self._joint_ids, torch.Tensor) else self._joint_ids
        self._position_targets = self._asset.data.joint_pos.torch[:, self._torch_joint_ids].clone()
        self._gravity_joint_ids = (
            [joint_id + self._asset.num_base_dofs for joint_id in range(self._asset.num_joints)]
            if isinstance(self._torch_joint_ids, slice)
            else self._torch_joint_ids + self._asset.num_base_dofs
        )
        self._workspace_lower = torch.tensor(cfg.workspace_lower, device=self.device, dtype=torch.float32)
        self._workspace_upper = torch.tensor(cfg.workspace_upper, device=self.device, dtype=torch.float32)
        if self._workspace_lower.shape != (self.action_dim,) or self._workspace_upper.shape != (self.action_dim,):
            raise ValueError(
                "workspace_lower and workspace_upper must each contain one value "
                f"per controlled joint ({self.action_dim})."
            )
        if torch.any(self._workspace_lower >= self._workspace_upper):
            raise ValueError("Every workspace_lower value must be less than workspace_upper.")

    def process_actions(self, actions: torch.Tensor) -> None:
        """Create one bounded position target from the measured joint pose."""
        super().process_actions(actions)
        target_delta = torch.clamp(
            self._processed_actions,
            min=-self.cfg.max_delta,
            max=self.cfg.max_delta,
        )
        current_position = self._asset.data.joint_pos.torch[:, self._torch_joint_ids]
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._torch_joint_ids]
        lower = torch.maximum(
            limits[..., 0] + self.cfg.joint_limit_margin,
            self._workspace_lower,
        )
        upper = torch.minimum(
            limits[..., 1] - self.cfg.joint_limit_margin,
            self._workspace_upper,
        )
        self._position_targets = torch.clamp(current_position + target_delta, min=lower, max=upper)
        self._processed_actions = self._position_targets
        self._raw_actions[:] = actions

    def apply_actions(self) -> None:
        """Apply the fixed target and gravity feedforward during every physics substep."""
        self._asset.set_joint_position_target_index(target=self.processed_actions, joint_ids=self._joint_ids)
        if self.cfg.gravity_compensation:
            gravity = self._asset.data.gravity_compensation_forces.torch[:, self._gravity_joint_ids]
            gravity = torch.where(torch.isfinite(gravity), gravity, torch.zeros_like(gravity))
            self._asset.set_joint_effort_target_index(target=gravity, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Initialize the held target from each sampled reset pose."""
        super().reset(env_ids)
        current_position = self._asset.data.joint_pos.torch[:, self._torch_joint_ids]
        if env_ids is None:
            self._position_targets[:] = current_position
        else:
            self._position_targets[env_ids] = current_position[env_ids]
        self._processed_actions = self._position_targets
