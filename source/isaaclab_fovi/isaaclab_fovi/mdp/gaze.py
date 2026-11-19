# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded tilt and pan actions for a foveated camera."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import torch
import warp as wp

from isaaclab.managers import ActionTerm
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp.memory import resolve_1d_indices, resolve_ids

from ..sensors.gaze_kernels import record_launch, torch_stream, warp_stream

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from ..sensors import FoveatedCamera
    from .gaze_cfg import GazeActionCfg


@wp.kernel(enable_backward=False)
def gaze_step_kernel(
    actions: wp.array(dtype=wp.vec2f),
    state: wp.array(dtype=wp.vec4f),
    angles_deg: wp.array(dtype=wp.vec2f),
    step_size: wp.float32,
    gaze_limit: wp.float32,
    step_dt: wp.float32,
    field_radius_deg: wp.float32,
):
    """Move each gaze by its clamped action and keep it within the limit; record the gaze, velocity and angles.

    The step starts from the camera's stored ``angles_deg``, which own the gaze. ``state`` records each gaze and its
    velocity, in field radii and field radii per second.
    """
    i = wp.tid()
    a = actions[i]
    previous = angles_deg[i] / field_radius_deg
    moved = previous + wp.vec2f(wp.clamp(a[0], -1.0, 1.0), wp.clamp(a[1], -1.0, 1.0)) * step_size
    current = moved * (gaze_limit / wp.max(wp.length(moved), gaze_limit))
    rate = (current - previous) / step_dt
    angles_deg[i] = current * field_radius_deg
    state[i] = wp.vec4f(current[0], current[1], rate[0], rate[1])


def gaze_state(env: ManagerBasedEnv, action_name: str = "gaze") -> torch.Tensor:
    """Gaze and its velocity, in field radii and field radii per second, shape (num_envs, 4)."""
    term: GazeAction = env.action_manager.get_term(action_name)
    return term.state


class GazeAction(ActionTerm):
    """Move gaze by clipped tilt and pan increments within a disk of permitted angles.

    Angles and increments are in units of the retina's field radius. Reset returns the gaze to the optical axis.
    The camera owns the angles. The action reports the gaze and its angular velocity after the last action or reset.
    """

    cfg: GazeActionCfg
    _asset: FoveatedCamera

    is_graphable: ClassVar[bool] = True
    """The gaze moves in :meth:`process_actions`, once per environment step; :meth:`apply_actions` does nothing, so
    physics may record the other terms' actions and its decimation loop into a CUDA graph."""

    def __init__(self, cfg: GazeActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.field_radius_deg = self._asset.sampling_coords.fov / 2
        self._raw: torch.Tensor | None = None
        """Gaze actions of the last step, as supplied to :meth:`process_actions`."""
        self._state = ProxyArray(wp.zeros(self.num_envs, dtype=wp.vec4f, device=self.device))
        """Gaze and its velocity as of the last step or reset, the gaze observation."""
        self._all_ids = wp.from_torch(torch.arange(self.num_envs, dtype=torch.int32, device=self.device))
        self._step_launch: tuple[tuple, wp.Launch] | None = None

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw if self._raw is not None else torch.zeros_like(self._state.torch[:, :2])

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._asset.gaze_angles.torch / self.field_radius_deg

    @property
    def state(self) -> torch.Tensor:
        """Gaze and its velocity, in field radii and field radii per second, shape (num_envs, 4)."""
        return self._state.torch

    def process_actions(self, actions: torch.Tensor):
        """Move the gaze by the supplied actions, shape (num_envs, 2).

        The launch reads ``actions`` in place. The action manager hands the term the same slice of its buffer each
        step, so the launch is recorded once; another buffer records a launch that reads it.
        """
        if actions.dtype != torch.float32:
            actions = actions.float()
        key = (actions.data_ptr(), actions.stride(), actions.device)
        if self._step_launch is None or self._step_launch[0] != key:
            launch = record_launch(
                gaze_step_kernel,
                self.num_envs,
                [
                    wp.from_torch(actions, dtype=wp.vec2f),
                    self._state,
                    self._asset.gaze_angles,
                    self.cfg.step_size,
                    self.cfg.gaze_limit,
                    self._env.step_dt,
                    self.field_radius_deg,
                ],
                device=self.device,
            )
            self._step_launch = (key, launch)
        self._raw = actions
        self._step_launch[1].launch(warp_stream(self.device))
        self._asset.set_gaze_angles(self._asset.gaze_angles.torch)

    def apply_actions(self):
        pass

    def reset(self, env_ids=None):
        """Return the selected cameras to the optical axis, at rest."""
        angles = self._asset.gaze_angles
        with torch_stream(self.device):
            if env_ids is None:
                angles.warp.zero_()
                self._state.warp.zero_()
            else:
                indices = resolve_1d_indices(resolve_ids(ids=env_ids, all_ids=self._all_ids))
                wp.indexedarray(angles.warp, indices).zero_()
                wp.indexedarray(self._state.warp, indices).zero_()
        self._asset.set_gaze_angles(angles.torch)
