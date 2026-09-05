# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import gymnasium as gym
import torch
from tensordict import TensorDict
from tensordict.utils import expand_as_right
from torchrl.data import Bounded, Categorical, Composite, MultiCategorical, TensorSpec, Unbounded
from torchrl.envs import EnvBase

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

    with contextlib.suppress(ImportError):
        from isaaclab_experimental.envs import DirectRLEnvWarp, ManagerBasedRLEnvWarp


class IsaacLabTorchRLWrapper(EnvBase):
    """Wraps around Isaac Lab environment for the TorchRL library.

    The wrapper implements :class:`~torchrl.envs.EnvBase` directly on the batched Isaac Lab environment
    instead of going through :class:`torchrl.envs.libs.gym.GymWrapper`, so it also accepts the Warp-based
    environments (which are not ``gymnasium`` subclasses) and keeps the tensordicts on the simulation device
    by default. Observation groups (e.g. ``"policy"``, ``"critic"``) become one :class:`~torchrl.data.Composite`
    each, and ``done``, ``terminated``, and ``truncated`` are separate spec keys.

    Isaac Lab resets finished environments inside :meth:`step` and returns the first observation of their new
    episodes. The ``"next"`` observation of a done transition is therefore taken from ``extras["final_obs"]``
    when the environment captures it (``cfg.compute_final_obs=True``), which keeps value bootstrapping on
    time-outs correct; otherwise it is NaN-filled following TorchRL's auto-reset convention. The ``"_reset"``
    requests TorchRL issues after a step with done environments (including the case where all of them
    finished) return the current observations without resetting again; only :meth:`reset` calls without a
    ``"_reset"`` mask reset the simulation.

    Episode statistics (``extras["log"]``) are per-batch scalars and are not part of the tensordicts; they
    remain available under ``env.unwrapped.extras["log"]``.

    .. caution::
        This class must be the last wrapper in the wrapper chain, since it does not follow the
        :class:`gymnasium.Wrapper` interface.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv | DirectRLEnv,
        device: str | None = None,
        clip_actions: float | None = None,
    ):
        """Initializes the wrapper.

        Args:
            env: The environment to wrap around.
            device: The device the returned tensordicts should live on. If ``None``, the
                wrapper stays on :attr:`env`'s device and performs no device conversion.
            clip_actions: The clipping value for actions. If ``None``, then no clipping is
                done. Only supported for :class:`gymnasium.spaces.Box` action spaces.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv`
                or :class:`DirectRLEnv`.
        """
        # NOTE: import here (not at module level) to avoid loading heavy env classes before Isaac Sim is initialized.
        from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

        try:
            from isaaclab_experimental.envs import DirectRLEnvWarp, ManagerBasedRLEnvWarp
        except ImportError:
            DirectRLEnvWarp = None
            ManagerBasedRLEnvWarp = None

        allowed_types = (ManagerBasedRLEnv, DirectRLEnv)
        if DirectRLEnvWarp is not None:
            allowed_types += (DirectRLEnvWarp,)
        if ManagerBasedRLEnvWarp is not None:
            allowed_types += (ManagerBasedRLEnvWarp,)

        if not isinstance(env.unwrapped, allowed_types):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv / DirectRLEnv / DirectRLEnvWarp /"
                f" ManagerBasedRLEnvWarp. Environment type: {type(env)}"
            )

        self.env = env
        self._clip_actions = clip_actions
        num_envs = self.unwrapped.num_envs

        super().__init__(
            device=device if device is not None else self.unwrapped.device,
            batch_size=torch.Size([num_envs]),
        )

        self.observation_spec = self._make_observation_spec(self.unwrapped.single_observation_space)
        self.action_spec = self._make_action_spec(self.unwrapped.single_action_space, clip_actions)
        self.reward_spec = Unbounded(shape=(*self.batch_size, 1), device=self.device)
        self.done_spec = Composite(
            {
                key: Categorical(2, dtype=torch.bool, shape=(*self.batch_size, 1), device=self.device)
                for key in ("done", "terminated", "truncated")
            },
            shape=self.batch_size,
        )
        # the wrapped environment is already running
        self.is_closed = False

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties
    """

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv | DirectRLEnvWarp | ManagerBasedRLEnvWarp:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Operations - EnvBase
    """

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Steps the environment with the ``"action"`` entry of ``tensordict``.

        Returns:
            The observation groups, ``reward``, ``terminated``, ``truncated`` and ``done`` of the transition.
        """
        actions = tensordict.get("action")
        if self._clip_actions is not None:
            actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        # Isaac Lab reuses these buffers across steps; TorchRL keeps every step's tensordict alive until stacked.
        rew = rew.reshape(*self.batch_size, 1).clone()
        terminated = terminated.reshape(*self.batch_size, 1).clone()
        truncated = truncated.reshape(*self.batch_size, 1).clone()
        if getattr(self.unwrapped.cfg, "is_finite_horizon", False):
            # finite-horizon tasks treat the time limit as terminal (same convention as the RSL-RL wrapper)
            terminated |= truncated
            truncated.zero_()
        done = terminated | truncated

        obs = TensorDict(obs_dict, batch_size=self.batch_size, device=self.device)
        out = self._terminal_observations(obs, extras, done)
        out["reward"] = rew
        out["terminated"] = terminated
        out["truncated"] = truncated
        out["done"] = done
        return out

    def _reset(self, tensordict: TensorDict | None = None, **kwargs) -> TensorDict:
        """Resets the environment and returns the initial observations.

        A ``"_reset"`` mask in ``tensordict`` does not reset anything: TorchRL issues it after a step with done
        environments, which Isaac Lab already reset inside :meth:`step`, so the current observations are returned.
        """
        if tensordict is not None and tensordict.get("_reset", None) is not None:
            obs_dict = self.unwrapped.obs_buf
        else:
            reset_kwargs = {key: kwargs[key] for key in ("seed", "options") if key in kwargs}
            obs_dict, _ = self.env.reset(**reset_kwargs)
        return TensorDict(obs_dict, batch_size=self.batch_size, device=self.device)

    def _set_seed(self, seed: int | None) -> None:
        """Seeds the environment. ``None`` leaves the random number generators untouched."""
        if seed is not None:
            self.unwrapped.seed(seed)

    def close(self, *, raise_if_closed: bool = True) -> None:  # noqa: D102
        if not self.is_closed:
            self.env.close()
        self.is_closed = True

    """
    Helper functions
    """

    def _terminal_observations(self, obs: TensorDict, extras: dict, done: torch.Tensor) -> TensorDict:
        """Replaces the post-reset observations of done environments with ``extras["final_obs"]`` when available.

        Without terminal observations, done rows are NaN-filled (zero-filled for non-floating-point observations)
        following TorchRL's auto-reset convention.
        """
        done = done.reshape(self.batch_size)
        if "final_obs" in extras:
            final_obs = TensorDict(extras["final_obs"], batch_size=self.batch_size, device=self.device)
            return obs.where(~done, final_obs)

        def _invalidate(value: torch.Tensor) -> torch.Tensor:
            fill = torch.full_like(value, torch.nan) if value.is_floating_point() else torch.zeros_like(value)
            return torch.where(expand_as_right(done, value), fill, value)

        return obs.apply(_invalidate)

    def _make_observation_spec(self, obs_space: gym.spaces.Dict) -> Composite:
        """Builds a :class:`~torchrl.data.Composite` mirroring Isaac Lab's observation groups."""
        return Composite(
            {group: self._gym_space_to_spec(space) for group, space in obs_space.spaces.items()},
            shape=self.batch_size,
        )

    def _gym_space_to_spec(self, space: gym.Space) -> TensorSpec | Composite:
        """Converts a per-environment gymnasium space into a batch-prefixed TorchRL spec.

        Raises:
            NotImplementedError: When the space is not a :class:`gymnasium.spaces.Box` or a
                :class:`gymnasium.spaces.Dict` of supported spaces.
        """
        if isinstance(space, gym.spaces.Dict):
            return Composite(
                {key: self._gym_space_to_spec(subspace) for key, subspace in space.spaces.items()},
                shape=self.batch_size,
            )
        if isinstance(space, gym.spaces.Box):
            shape = (*self.batch_size, *space.shape)
            low = torch.as_tensor(space.low)
            high = torch.as_tensor(space.high)
            if bool(torch.isfinite(low).all()) and bool(torch.isfinite(high).all()):
                return Bounded(
                    low=low.expand(shape), high=high.expand(shape), shape=shape, dtype=low.dtype, device=self.device
                )
            return Unbounded(shape=shape, dtype=low.dtype, device=self.device)
        raise NotImplementedError(f"Space type {type(space)} is not supported by {type(self).__name__}.")

    def _make_action_spec(self, action_space: gym.Space, clip_actions: float | None) -> TensorSpec:
        """Builds the action spec, narrowed to ``[-clip_actions, clip_actions]`` when clipping is enabled.

        Raises:
            ValueError: When ``clip_actions`` is set for a non-:class:`gymnasium.spaces.Box` action space.
            NotImplementedError: When the action space type is not supported.
        """
        if isinstance(action_space, gym.spaces.Box):
            if clip_actions is not None:
                shape = (*self.batch_size, *action_space.shape)
                return Bounded(low=-clip_actions, high=clip_actions, shape=shape, device=self.device)
            return self._gym_space_to_spec(action_space)
        if clip_actions is not None:
            raise ValueError(f"Action clipping is only supported for Box action spaces, got {type(action_space)}.")
        if isinstance(action_space, gym.spaces.Discrete):
            return Categorical(int(action_space.n), shape=(*self.batch_size, 1), dtype=torch.int64, device=self.device)
        if isinstance(action_space, gym.spaces.MultiDiscrete):
            nvec = action_space.nvec.tolist()
            return MultiCategorical(nvec, shape=(*self.batch_size, len(nvec)), dtype=torch.int64, device=self.device)
        raise NotImplementedError(f"Action space type {type(action_space)} is not supported by {type(self).__name__}.")
