# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the SO101 keyboard letter-typing task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg

    from .commands import LetterTypingCommand


def _slots_onehot(slots: torch.Tensor, num_keys: int) -> torch.Tensor:
    """One-hot a ``(num_envs, max_len)`` slot-id buffer (``-1`` marks empty) into ``(num_envs, max_len * num_keys)``.

    Empty/padding positions (slot ``-1``) become all-zero rows, so they contribute nothing.
    """
    valid = slots >= 0
    onehot = torch.nn.functional.one_hot(slots.clamp(min=0), num_classes=num_keys).to(torch.float)
    onehot = onehot * valid.unsqueeze(-1).to(onehot.dtype)
    return onehot.reshape(slots.shape[0], -1)


def target_keys_onehot(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Target word as a one-hot sequence over key slots, shape ``(num_envs, max_len * num_keys)``.

    Each sequence position holds a one-hot of that target key (all-zero past the word's length). Pair with
    the perception key-position map (same global slot order) so a gather/attention actor can recover where
    each target key is.
    """
    command: LetterTypingCommand = env.command_manager.get_term(command_name)  # type: ignore
    return _slots_onehot(command.target, command.num_keys)


def typed_keys_onehot(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Typed buffer as a one-hot sequence over key slots, shape ``(num_envs, max_len * num_keys)``.

    Each sequence position holds a one-hot of the key typed there so far (all-zero for not-yet-typed
    positions). Comparing it against :func:`target_keys_onehot` yields the typing progress and any
    mistake the agent must backspace.
    """
    command: LetterTypingCommand = env.command_manager.get_term(command_name)  # type: ignore
    return _slots_onehot(command.typed, command.num_keys)


class key_positions_b(ManagerTermBase):
    """Per-key keyboard positions ``(x, y, z)`` [m] in the base asset's root frame, in global slot order.

    A gather map is built once from the keyboard articulation's ``(instance, body)`` layout into the
    canonical global key-slot order (``slot = part * dof + local``), mirroring the mapping in
    :class:`~isaaclab_tasks.contrib.keyboard.mdp.commands.LetterTypingCommand`. This makes the
    observation layout-stable: column ``s`` always refers to the same logical key slot regardless of how
    the keyboard is partitioned into articulation instances. Inactive slots are zero-padded, so a
    keyboard with fewer than :attr:`num_slots` real keys fills only its active subset of columns and the
    remainder stays ``0``.

    The map is agnostic to the partition mode:

    * ``single`` -> one instance per env; key body ``key_{slot:03d}``.
    * ``fixed_dof`` -> ``parts/part_*`` instances per env; key body ``key_{local:03d}``.

    Args (from ``cfg.params``):
        command_name: Name of the typing command that owns the canonical key-slot map. Defaults to ``typing``.
        base_asset_cfg: Scene entity providing the reference root frame. Defaults to ``SceneEntityCfg("robot")``.
        active_slots: Global key-slot ids that hold a real key; all other slots are zero-padded. ``None``
            keeps every slot that mapped to a key body.

    Returns (from :meth:`__call__`):
        Tensor of shape ``(num_envs, num_slots * 3)`` with per-slot key positions [m] in the base frame,
        flattened in global slot order and with inactive slots zeroed.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        command: LetterTypingCommand = env.command_manager.get_term(cfg.params.get("command_name", "typing"))  # type: ignore
        self.keyboard = command.keyboard
        self.num_slots = command.num_keys
        self._inst_idx = command._inst_idx
        self._body_idx = command._body_col_idx

        # Zero-padding mask: every gathered slot maps to a real key body, so keep them all unless
        # active_slots restricts the observation to a subset of slots (the rest stay zeroed).
        active_slots: tuple[int, ...] | None = cfg.params.get("active_slots", None)  # type: ignore
        if active_slots is None:
            active = torch.ones(self.num_envs, self.num_slots, dtype=torch.bool, device=self.device)
        else:
            active = torch.zeros(self.num_envs, self.num_slots, dtype=torch.bool, device=self.device)
            active[:, torch.as_tensor(tuple(active_slots), dtype=torch.long, device=self.device)] = True
        self._active = active.unsqueeze(-1).float()  # (num_envs, num_slots, 1)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str = "typing",
        base_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        active_slots: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        base: Articulation = env.scene[base_asset_cfg.name]
        # (num_envs, num_slots, 3) key body world positions gathered into global slot order.
        key_pos_w = self.keyboard.data.body_pos_w.torch[self._inst_idx, self._body_idx]
        # broadcast the base root pose over slots and transform world -> base frame (position only).
        root_pos_w = base.data.root_link_pos_w.torch.unsqueeze(1).expand(-1, self.num_slots, -1).reshape(-1, 3)
        root_quat_w = base.data.root_link_quat_w.torch.unsqueeze(1).expand(-1, self.num_slots, -1).reshape(-1, 4)
        key_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, key_pos_w.reshape(-1, 3))
        key_pos_b = key_pos_b.view(env.num_envs, self.num_slots, 3) * self._active
        return key_pos_b.reshape(env.num_envs, -1)
