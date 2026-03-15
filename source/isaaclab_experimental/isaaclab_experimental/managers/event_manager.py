# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event manager for orchestrating operations based on different simulation events (experimental, Warp-first).

This module mirrors :mod:`isaaclab.managers.event_manager` but removes torch ops from hot paths to enable
CUDA-graph-friendly execution for modes that can be captured (notably ``interval``).

Key differences from the stable manager:
- ``interval`` and ``reset`` modes are **mask-based** internally and implemented using Warp kernels.
- No ``torch.rand`` / ``nonzero`` / tensor allocations in the ``interval`` apply path.

Event term signature for Warp-first interval/reset modes:
    ``func(env, env_mask_wp, **params) -> None``

Other modes (e.g. ``prestartup``, ``startup``) are called using the stable convention:
    ``func(env, env_ids, **params) -> None``
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Sequence

import torch
import warp as wp
from prettytable import PrettyTable

from .manager_base import ManagerBase
from .manager_term_cfg import EventTermCfg

logger = logging.getLogger(__name__)


@wp.kernel
def _interval_init_per_env(
    time_left: wp.array(dtype=wp.float32),
    rng_state: wp.array(dtype=wp.uint32),
    lower: wp.float32,
    upper: wp.float32,
):
    env_id = wp.tid()
    s = rng_state[env_id]
    time_left[env_id] = wp.randf(s, lower, upper)
    rng_state[env_id] = s


@wp.kernel
def _interval_init_global(
    time_left: wp.array(dtype=wp.float32),
    rng_state: wp.array(dtype=wp.uint32),
    lower: wp.float32,
    upper: wp.float32,
):
    # single element
    s = rng_state[0]
    time_left[0] = wp.randf(s, lower, upper)
    rng_state[0] = s


@wp.kernel
def _interval_step_per_env(
    time_left: wp.array(dtype=wp.float32),
    rng_state: wp.array(dtype=wp.uint32),
    trigger_mask: wp.array(dtype=wp.bool),
    dt: wp.float32,
    lower: wp.float32,
    upper: wp.float32,
):
    env_id = wp.tid()
    t = time_left[env_id] - dt
    if t < wp.float32(1.0e-6):
        trigger_mask[env_id] = True
        s = rng_state[env_id]
        time_left[env_id] = wp.randf(s, lower, upper)
        rng_state[env_id] = s
    else:
        trigger_mask[env_id] = False
        time_left[env_id] = t


@wp.kernel
def _interval_step_global(
    time_left: wp.array(dtype=wp.float32),
    rng_state: wp.array(dtype=wp.uint32),
    trigger_flag: wp.array(dtype=wp.bool),
    dt: wp.float32,
    lower: wp.float32,
    upper: wp.float32,
):
    t = time_left[0] - dt
    if t < wp.float32(1.0e-6):
        trigger_flag[0] = True
        s = rng_state[0]
        time_left[0] = wp.randf(s, lower, upper)
        rng_state[0] = s
    else:
        trigger_flag[0] = False
        time_left[0] = t


@wp.kernel
def _interval_reset_selected(
    env_mask: wp.array(dtype=wp.bool),
    time_left: wp.array(dtype=wp.float32),
    rng_state: wp.array(dtype=wp.uint32),
    lower: wp.float32,
    upper: wp.float32,
):
    env_id = wp.tid()
    if env_mask[env_id]:
        s = rng_state[env_id]
        time_left[env_id] = wp.randf(s, lower, upper)
        rng_state[env_id] = s


@wp.kernel
def _seed_global_rng_from_env_rng(
    env_rng_state: wp.array(dtype=wp.uint32),
    global_rng_state: wp.array(dtype=wp.uint32),
):
    global_rng_state[0] = wp.rand_init(wp.int32(env_rng_state[0]), wp.int32(0))


@wp.kernel
def _reset_compute_valid_mask(
    in_mask: wp.array(dtype=wp.bool),
    last_triggered_step: wp.array(dtype=wp.int32),
    triggered_once: wp.array(dtype=wp.bool),
    out_mask: wp.array(dtype=wp.bool),
    global_step_count_buf: wp.array(dtype=wp.int32),
    min_step_count: wp.int32,
):
    env_id = wp.tid()
    if not in_mask[env_id]:
        out_mask[env_id] = False
        return

    global_step_count = global_step_count_buf[0]
    if min_step_count == wp.int32(0):
        out_mask[env_id] = True
        last_triggered_step[env_id] = global_step_count
        triggered_once[env_id] = True
        return

    last = last_triggered_step[env_id]
    once = triggered_once[env_id]
    steps_since = global_step_count - last
    valid = steps_since >= min_step_count
    # Trigger at least once at the start (matching stable behavior).
    valid = valid or ((last == wp.int32(0)) and (not once))
    out_mask[env_id] = valid
    if valid:
        last_triggered_step[env_id] = global_step_count
        triggered_once[env_id] = True


class EventManager(ManagerBase):
    """Manager for orchestrating operations based on different simulation events (Warp-first for interval/reset)."""

    def __init__(self, cfg: object, env):
        # create buffers to parse and store terms
        self._mode_term_names: dict[str, list[str]] = {}
        self._mode_term_cfgs: dict[str, list[EventTermCfg]] = {}
        self._mode_class_term_cfgs: dict[str, list[EventTermCfg]] = {}

        # Warp buffers for interval/reset modes (populated in _prepare_terms)
        self._interval_term_time_left_wp: list[wp.array] = []
        self._interval_term_ranges: list[tuple[float, float]] = []
        self._interval_term_is_global: list[bool] = []
        # Scalar RNG state for global interval timers (allocated lazily if needed).
        self._interval_global_rng_state_wp: wp.array | None = None

        self._reset_term_last_triggered_step_wp: list[wp.array] = []
        self._reset_term_triggered_once_wp: list[wp.array] = []

        super().__init__(cfg, env)

        # persistent scratch mask for per-term interval/reset triggering (must be stable pointer for capture)
        self._scratch_term_mask_wp = wp.zeros((self.num_envs,), dtype=wp.bool, device=self.device)

        # scratch scalar flag & broadcast view for global interval triggering (no per-term masks)
        self._scratch_interval_trigger_flag_wp = wp.zeros((1,), dtype=wp.bool, device=self.device)
        self._scratch_interval_trigger_mask_view_wp = wp.array(
            ptr=self._scratch_interval_trigger_flag_wp.ptr,
            dtype=wp.bool,
            shape=(self.num_envs,),
            strides=(0,),
            capacity=self._scratch_interval_trigger_flag_wp.capacity,
            device=self._scratch_interval_trigger_flag_wp.device,
            copy=False,
        )

    def __str__(self) -> str:
        msg = f"<EventManager> contains {len(self._mode_term_names)} active terms.\n"
        for mode in self._mode_term_names:
            table = PrettyTable()
            table.title = f"Active Event Terms in Mode: '{mode}'"
            if mode == "interval":
                table.field_names = ["Index", "Name", "Interval time range (s)"]
                table.align["Name"] = "l"
                for index, (name, cfg) in enumerate(zip(self._mode_term_names[mode], self._mode_term_cfgs[mode])):
                    table.add_row([index, name, cfg.interval_range_s])
            else:
                table.field_names = ["Index", "Name"]
                table.align["Name"] = "l"
                for index, name in enumerate(self._mode_term_names[mode]):
                    table.add_row([index, name])
            msg += table.get_string()
            msg += "\n"
        return msg

    @property
    def active_terms(self) -> dict[str, list[str]]:
        return self._mode_term_names

    @property
    def available_modes(self) -> list[str]:
        return list(self._mode_term_names.keys())

    def set_term_cfg(self, term_name: str, cfg: EventTermCfg):
        term_found = False
        for mode, terms in self._mode_term_names.items():
            if term_name in terms:
                self._mode_term_cfgs[mode][terms.index(term_name)] = cfg
                term_found = True
                break
        if not term_found:
            raise ValueError(f"Event term '{term_name}' not found.")

    def get_term_cfg(self, term_name: str) -> EventTermCfg:
        for mode, terms in self._mode_term_names.items():
            if term_name in terms:
                return self._mode_term_cfgs[mode][terms.index(term_name)]
        raise ValueError(f"Event term '{term_name}' not found.")

    def reset(
        self,
        env_ids: Sequence[int] | slice | torch.Tensor | wp.array | None = None,
        *,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> dict[str, float]:
        # Mask-first path: captured callers must provide env_mask.
        if env_mask is None or not isinstance(env_mask, wp.array):
            # Keep all id->mask resolution strictly outside capture.
            if wp.get_device().is_capturing:
                raise RuntimeError(
                    "EventManager.reset requires env_mask(wp.array[bool]) during capture. "
                    "Do not pass env_ids on captured paths."
                )
            env_mask = self._env.resolve_env_mask(env_ids=env_ids, env_mask=env_mask)

        # reset class terms (mask-based)
        for mode_cfg in self._mode_class_term_cfgs.values():
            for term_cfg in mode_cfg:
                term_cfg.func.reset(env_mask=env_mask)

        # reset interval timers for non-global interval events
        if "interval" in self._mode_term_cfgs:
            for i, term_cfg in enumerate(self._mode_term_cfgs["interval"]):
                if term_cfg.is_global_time:
                    continue
                lower, upper = self._interval_term_ranges[i]
                wp.launch(
                    kernel=_interval_reset_selected,
                    dim=self.num_envs,
                    inputs=[
                        env_mask,
                        self._interval_term_time_left_wp[i],
                        self._env.rng_state_wp,
                        float(lower),
                        float(upper),
                    ],
                    device=self.device,
                )
        return {}

    def apply(
        self,
        mode: str,
        env_ids: Sequence[int] | slice | torch.Tensor | wp.array | None = None,
        dt: float | None = None,
        global_env_step_count: wp.array | None = None,
        *,
        env_mask_wp: wp.array | None = None,
    ):
        if mode not in self._mode_term_names:
            logger.warning(f"Event mode '{mode}' is not defined. Skipping event.")
            return

        # SceneEntityCfg-dependent term params should be resolved before entering captured event paths.
        if not self._is_scene_entities_resolved:
            if wp.get_device().is_capturing:
                raise RuntimeError(
                    "EventManager terms are unresolved during CUDA graph capture. "
                    "Resolve terms before entering captured event paths."
                )
            if self._env.sim.is_playing():
                self._resolve_terms_callback(None)

        if mode == "interval":
            if dt is None:
                raise ValueError(f"Event mode '{mode}' requires the time-step of the environment.")
            if env_ids is not None:
                raise ValueError(
                    f"Event mode '{mode}' does not require environment indices. This is an undefined behavior."
                )
            self._apply_interval(float(dt))
            return

        if mode == "reset":
            if global_env_step_count is None:
                raise ValueError(f"Event mode '{mode}' requires the total number of environment steps to be provided.")
            if env_mask_wp is None:
                if wp.get_device().is_capturing:
                    raise ValueError(
                        f"Event mode '{mode}' requires the environment mask to be provided when capturing."
                    )
                env_mask_wp = self._env.resolve_env_mask(env_ids=env_ids)
            self._apply_reset(env_mask_wp, global_env_step_count)
            return

        # other modes keep the stable convention (env_ids forwarded)
        for term_cfg in self._mode_term_cfgs[mode]:
            term_cfg.func(self._env, env_ids, **term_cfg.params)

    def _apply_interval(self, dt: float) -> None:
        if self._env.rng_state_wp is None:
            raise RuntimeError("EventManager._apply_interval: env.rng_state_wp is not initialized.")

        # iterate over all the interval terms (fixed list; captured graph-friendly)
        for i, term_cfg in enumerate(self._mode_term_cfgs["interval"]):
            lower, upper = self._interval_term_ranges[i]
            if self._interval_term_is_global[i]:
                if self._interval_global_rng_state_wp is None:
                    raise RuntimeError(
                        "EventManager._apply_interval: _interval_global_rng_state_wp is not initialized."
                    )
                # update scalar time_left and scalar flag (mask is a broadcast view of the flag)
                wp.launch(
                    kernel=_interval_step_global,
                    dim=1,
                    inputs=[
                        self._interval_term_time_left_wp[i],
                        self._interval_global_rng_state_wp,
                        self._scratch_interval_trigger_flag_wp,
                        float(dt),
                        float(lower),
                        float(upper),
                    ],
                    device=self.device,
                )
                term_cfg.func(self._env, self._scratch_interval_trigger_mask_view_wp, **term_cfg.params)
            else:
                wp.launch(
                    kernel=_interval_step_per_env,
                    dim=self.num_envs,
                    inputs=[
                        self._interval_term_time_left_wp[i],
                        self._env.rng_state_wp,
                        self._scratch_term_mask_wp,
                        float(dt),
                        float(lower),
                        float(upper),
                    ],
                    device=self.device,
                )
                term_cfg.func(self._env, self._scratch_term_mask_wp, **term_cfg.params)

    def _apply_reset(self, env_mask_wp: wp.array, global_env_step_count_wp: wp.array) -> None:
        if self._scratch_term_mask_wp is None:
            raise RuntimeError("EventManager._apply_reset: _scratch_term_mask_wp is not initialized.")

        # iterate over all the reset terms
        for index, term_cfg in enumerate(self._mode_term_cfgs["reset"]):
            min_step_count = int(term_cfg.min_step_count_between_reset)
            wp.launch(
                kernel=_reset_compute_valid_mask,
                dim=self.num_envs,
                inputs=[
                    env_mask_wp,
                    self._reset_term_last_triggered_step_wp[index],
                    self._reset_term_triggered_once_wp[index],
                    self._scratch_term_mask_wp,
                    global_env_step_count_wp,
                    int(min_step_count),
                ],
                device=self.device,
            )
            term_cfg.func(self._env, self._scratch_term_mask_wp, **term_cfg.params)

    def _prepare_terms(self):
        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()

        # iterate over all the terms
        for term_name, term_cfg in cfg_items:
            if term_cfg is None:
                continue
            if not isinstance(term_cfg, EventTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type EventTermCfg. Received:"
                    f" '{type(term_cfg)}'."
                )

            if term_cfg.mode != "reset" and term_cfg.min_step_count_between_reset != 0:
                logger.warning(
                    f"Event term '{term_name}' has 'min_step_count_between_reset' set to a non-zero value"
                    " but the mode is not 'reset'. Ignoring the 'min_step_count_between_reset' value."
                )

            # resolve common parameters
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=2)

            # check if mode is pre-startup and scene replication is enabled
            if term_cfg.mode == "prestartup" and self._env.scene.cfg.replicate_physics:
                raise RuntimeError(
                    "Scene replication is enabled, which may affect USD-level randomization."
                    " When assets are replicated, their properties are shared across instances,"
                    " potentially leading to unintended behavior."
                    " For stable USD-level randomization, please disable scene replication"
                    " by setting 'replicate_physics' to False in 'InteractiveSceneCfg'."
                )

            # for prestartup callable class terms, initialize early (stable behavior)
            if inspect.isclass(term_cfg.func) and term_cfg.mode == "prestartup":
                logger.info(f"Initializing term '{term_name}' with class '{term_cfg.func.__name__}'.")
                term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)

            # ensure mode buckets exist
            if term_cfg.mode not in self._mode_term_names:
                self._mode_term_names[term_cfg.mode] = []
                self._mode_term_cfgs[term_cfg.mode] = []
                self._mode_class_term_cfgs[term_cfg.mode] = []
            # add term name and parameters
            self._mode_term_names[term_cfg.mode].append(term_name)
            self._mode_term_cfgs[term_cfg.mode].append(term_cfg)

            if inspect.isclass(term_cfg.func):
                self._mode_class_term_cfgs[term_cfg.mode].append(term_cfg)

            # per-mode Warp buffers
            if term_cfg.mode == "interval":
                if term_cfg.interval_range_s is None:
                    raise ValueError(
                        f"Event term '{term_name}' has mode 'interval' but 'interval_range_s' is not specified."
                    )
                lower, upper = term_cfg.interval_range_s
                self._interval_term_ranges.append((float(lower), float(upper)))

                if term_cfg.is_global_time:
                    # allocate and seed scalar global RNG state if needed (avoid consuming env0 RNG stream)
                    if self._interval_global_rng_state_wp is None:
                        if self._env.rng_state_wp is None:
                            raise RuntimeError("EventManager._prepare_terms: env.rng_state_wp is not initialized.")
                        self._interval_global_rng_state_wp = wp.zeros((1,), dtype=wp.uint32, device=self.device)
                        wp.launch(
                            kernel=_seed_global_rng_from_env_rng,
                            dim=1,
                            inputs=[self._env.rng_state_wp, self._interval_global_rng_state_wp],
                            device=self.device,
                        )
                    time_left = wp.zeros((1,), dtype=wp.float32, device=self.device)
                    wp.launch(
                        kernel=_interval_init_global,
                        dim=1,
                        inputs=[time_left, self._interval_global_rng_state_wp, float(lower), float(upper)],
                        device=self.device,
                    )
                    self._interval_term_time_left_wp.append(time_left)
                    self._interval_term_is_global.append(True)
                else:
                    time_left = wp.zeros((self.num_envs,), dtype=wp.float32, device=self.device)
                    wp.launch(
                        kernel=_interval_init_per_env,
                        dim=self.num_envs,
                        inputs=[time_left, self._env.rng_state_wp, float(lower), float(upper)],
                        device=self.device,
                    )
                    self._interval_term_time_left_wp.append(time_left)
                    self._interval_term_is_global.append(False)

            elif term_cfg.mode == "reset":
                if term_cfg.min_step_count_between_reset < 0:
                    raise ValueError(
                        f"Event term '{term_name}' has mode 'reset' but 'min_step_count_between_reset' is"
                        f" negative: {term_cfg.min_step_count_between_reset}. Please provide a non-negative value."
                    )
                # per-env last-trigger bookkeeping (Warp)
                self._reset_term_last_triggered_step_wp.append(
                    wp.zeros((self.num_envs,), dtype=wp.int32, device=self.device)
                )
                self._reset_term_triggered_once_wp.append(wp.zeros((self.num_envs,), dtype=wp.bool, device=self.device))
