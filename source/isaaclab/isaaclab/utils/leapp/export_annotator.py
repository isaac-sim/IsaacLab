# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export annotations for Isaac Lab policies using proxy-based patching.

Observation and action annotation share a unified dedup cache so that a
state property (e.g. ``joint_pos``) read by both an observation term and
an action term resolves to one LEAPP input edge.

- Observation term functions see an ``_EnvProxy`` whose scene returns
  ``_EntityProxy`` objects with annotating data proxies.

- Action terms have their ``_asset`` attribute replaced with an
  _ArticulationWriteProxy that intercepts ``_leapp_semantics``-decorated
  write methods **and** routes ``.data`` reads through the same annotating
  data proxy used by observations.

Cache lifecycle (assuming single-env play-mode export):

    compute()                clear cache → obs terms populate cache
    policy inference         TracedTensors propagate through NN
    process_action()         register_buffer for raw_actions
    apply_action() [tracing] reuse cached TracedTensors for state reads,
                             capture write outputs, call output_tensors(),
                             then clear cache
    apply_action() [decim.]  clear cache → fresh reads for simulation
    ...
    compute()                clear cache → fresh reads for next obs
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import torch
from leapp import annotate
from leapp.utils.tensor_description import TensorSemantics

from isaaclab.assets.articulation.base_articulation import BaseArticulation
from isaaclab.managers import ManagerTermBase

from .leapp_semantics import select_element_names
from .proxy import _ArticulationWriteProxy, _DataProxy, _EnvProxy, _ManagerTermProxy
from .utils import (
    TracedProxyArray,
    build_command_connection,
    build_write_connection,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


VARIABLE_IMPEDANCE_MODES = frozenset({"variable", "variable_kp"})


# ══════════════════════════════════════════════════════════════════
# ExportPatcher
# ══════════════════════════════════════════════════════════════════


class ExportPatcher:
    """Unified patcher that annotates observation inputs and action outputs for LEAPP export.

    Observation-side property semantics are resolved lazily inside
    ``_DataProxy`` by combining:

    - the concrete runtime getter from the backend data class
    - the nearest ``_leapp_semantics`` metadata found while walking the MRO

    This lets backends override property implementations without duplicating
    decorators from the abstract API.

    - The observation proxy chain (``_EnvProxy`` → ``_SceneProxy`` →
      ``_EntityProxy`` → ``_DataProxy``) for state reads
      by observation term functions.
    - The ``_ArticulationWriteProxy`` on each action term, which intercepts
      target writes **and** routes ``.data`` reads through the same
      ``_DataProxy`` / cache.

    """

    def __init__(self, export_method: str, required_obs_groups: set[str] | None = None):
        """Initialize the export patcher.

        Args:
            export_method: LEAPP export backend passed to
                :func:`annotate.output_tensors`.
            required_obs_groups: Observation groups that should be patched, or
                ``None`` to patch all groups.
        """
        self.task_name: str | None = None
        self.export_method = export_method
        self.required_obs_groups = required_obs_groups
        self._annotated_tensor_cache: dict[tuple[int, str], TracedProxyArray] = {}
        self._data_property_resolution_cache: dict[tuple[type, str], tuple[Callable, object] | None] = {}
        self._write_method_resolution_cache: dict[
            tuple[type, str], tuple[Callable, object, inspect.Signature] | None
        ] = {}
        self._action_output_cache: list[TensorSemantics] = []
        self._captured_write_term_names: set[str] = set()
        self._fallback_term_names: set[str] = set()
        self._pending_action_output_export: bool = False
        self._uses_last_action_state: bool = False
        self._action_term_scene_keys: dict[str, str] = {}

    def setup(self, env):
        """Patch the environment in place for LEAPP-aware export.

        Args:
            env: Wrapped manager-based environment whose unwrapped instance
                should be patched.
        """
        unwrapped = env.env.unwrapped
        task_name = str(unwrapped.spec.id)
        self.task_name = task_name

        proxy_env = _EnvProxy(
            unwrapped,
            task_name,
            self._data_property_resolution_cache,
            self._annotated_tensor_cache,
        )

        self._disable_training_managers(unwrapped)
        self._patch_observation_manager(unwrapped.observation_manager, proxy_env)
        self._patch_history_buffers(unwrapped.observation_manager)
        self._patch_action_manager(
            unwrapped.action_manager,
            self._annotated_tensor_cache,
        )

    # ── Disable training-only managers ─────────────────────────────

    @staticmethod
    def _disable_training_managers(unwrapped):
        """Replace training-only manager methods with no-ops.

        During export the curriculum, reward, termination, and recorder
        managers serve no purpose.  Disabling them avoids side-effect
        crashes (e.g. ADR curriculum terms accessing nullified noise
        configs) and removes unnecessary computation.

        Args:
            unwrapped: Unwrapped environment whose training-only managers
                should be disabled.
        """
        num_envs = unwrapped.num_envs
        device = unwrapped.device
        _zero_reward = torch.zeros(num_envs, device=device)
        _no_termination = torch.zeros(num_envs, dtype=torch.bool, device=device)

        def _noop_curriculum(env_ids=None):
            return None

        def _zero_reward_compute(dt):
            return _zero_reward

        def _no_termination_compute():
            return _no_termination

        def _noop(*args, **kwargs):
            return None

        if hasattr(unwrapped, "curriculum_manager"):
            unwrapped.curriculum_manager.compute = _noop_curriculum

        if hasattr(unwrapped, "reward_manager"):
            unwrapped.reward_manager.compute = _zero_reward_compute

        if hasattr(unwrapped, "termination_manager"):
            unwrapped.termination_manager.compute = _no_termination_compute

        if hasattr(unwrapped, "recorder_manager"):
            rm = unwrapped.recorder_manager

            rm.record_pre_step = _noop
            rm.record_post_step = _noop
            rm.record_pre_reset = _noop
            rm.record_post_reset = _noop
            rm.record_post_physics_decimation_step = _noop

    @staticmethod
    def _resolve_scene_entity_key(scene, entity: Any) -> str | None:
        """Return the scene dictionary key for an entity.

        Args:
            scene: Scene object that stores entity dictionaries.
            entity: Entity instance to locate.

        Returns:
            The scene key for ``entity`` if found, otherwise ``None``.
        """
        for attr_value in vars(scene).values():
            if not isinstance(attr_value, dict):
                continue
            for key, candidate in attr_value.items():
                if candidate is entity:
                    return key
        return None

    # ── Observation manager patches ───────────────────────────────

    def _patch_history_buffers(self, obs_manager):
        """Patch history-enabled observation buffers to export as LEAPP state.

        Args:
            obs_manager: Observation manager whose history buffers should be
                wrapped.
        """
        history_buffers = getattr(obs_manager, "_group_obs_term_history_buffer", {})
        term_names_by_group = getattr(obs_manager, "_group_obs_term_names", {})

        for group_name, term_cfgs in obs_manager._group_obs_term_cfgs.items():
            if self.required_obs_groups is not None and group_name not in self.required_obs_groups:
                continue
            group_buffers = history_buffers.get(group_name, {})
            group_term_names = term_names_by_group.get(group_name, [])

            for index, term_cfg in enumerate(term_cfgs):
                history_length = getattr(term_cfg, "history_length", 0) or 0
                if history_length <= 0:
                    continue

                if index >= len(group_term_names):
                    continue

                term_name = group_term_names[index]
                circular_buffer = group_buffers.get(term_name)
                if circular_buffer is None:
                    continue

                state_name = f"h_{group_name}_{term_name}"
                self._patch_history_buffer_append(circular_buffer, state_name)

    def _patch_history_buffer_append(self, circular_buffer, state_name: str):
        """Wrap ``_append`` so history buffers become explicit LEAPP state.

        Args:
            circular_buffer: Circular buffer instance to patch.
            state_name: LEAPP state tensor name for the buffer contents.
        """
        if hasattr(circular_buffer, "_leapp_original_append"):
            return

        task_name = self.task_name
        original_append = circular_buffer._append

        def patched_append(data: torch.Tensor):
            """Annotate history buffer updates as LEAPP state transitions.

            Args:
                data: New observation slice appended to the buffer.

            Returns:
                ``None``.
            """
            if circular_buffer._buffer is not None:
                circular_buffer._buffer = annotate.state_tensors(task_name, {state_name: circular_buffer._buffer})

            original_append(data)

            if circular_buffer._buffer is not None:
                circular_buffer._buffer = annotate.update_state(task_name, {state_name: circular_buffer._buffer})

        circular_buffer._leapp_original_append = original_append
        circular_buffer._append = patched_append

    def _patch_observation_manager(self, obs_manager, proxy_env):
        """Patch observation terms to use annotating proxies and disable noise.

        Args:
            obs_manager: Observation manager instance to patch.
            proxy_env: Proxy environment routed into observation terms.
        """
        for group_name, term_cfgs in obs_manager._group_obs_term_cfgs.items():
            if self.required_obs_groups is not None and group_name not in self.required_obs_groups:
                continue
            for term_cfg in term_cfgs:
                original_func = term_cfg.func
                func_name = getattr(original_func, "__name__", None)

                if func_name == "last_action":
                    self._uses_last_action_state = True
                    term_cfg.func = self._wrap_last_action(original_func)
                elif func_name == "generated_commands":
                    term_cfg.func = self._wrap_generated_commands(original_func, term_cfg)
                else:
                    term_cfg.func = self._wrap_with_proxy(original_func, proxy_env)

                term_cfg.noise = None

        original_compute = obs_manager.compute
        cache = self._annotated_tensor_cache

        def patched_compute(*args, **kwargs):
            """Clear the tensor dedup cache once per full observation pass."""
            cache.clear()
            return original_compute(*args, **kwargs)

        obs_manager.compute = patched_compute

    # ── Action manager patches ────────────────────────────────────

    def _patch_action_manager(self, action_manager, cache):
        """Patch action terms with write/read proxies and manager hooks.

        Args:
            action_manager: Action manager instance to patch.
            cache: Shared tensor dedup cache for annotated state reads.
        """
        assert self.task_name is not None
        scene = action_manager._env.scene
        for term_name, term in action_manager._terms.items():
            asset = getattr(term, "_asset", None)
            if isinstance(asset, BaseArticulation):
                real_asset: BaseArticulation = asset
                scene_key = self._resolve_scene_entity_key(scene, real_asset) or "ego"
                data_proxy = _DataProxy(
                    real_asset.data,
                    scene_key,
                    self.task_name,
                    self._data_property_resolution_cache,
                    cache,
                    input_name_resolver=lambda prop_name, k=scene_key: f"{k}_{prop_name}",
                )
                term._asset = _ArticulationWriteProxy(
                    real_asset=real_asset,
                    entity_name=scene_key,
                    term_name=term_name,
                    output_cache=self._action_output_cache,
                    method_resolution_cache=self._write_method_resolution_cache,
                    captured_write_term_names=self._captured_write_term_names,
                    data_proxy=data_proxy,
                )
                self._action_term_scene_keys[term_name] = scene_key

        self._patch_action_manager_methods(action_manager)

    def _patch_action_manager_methods(self, action_manager):
        """Patch ``process_action`` and ``apply_action`` on the action manager instance.

        ``process_action`` registers raw_action buffers for LEAPP tracing and
        preserves the action tensor clone.

        ``apply_action`` coordinates the cache and output lifecycle:

        - **Tracing pass** (first ``apply_action`` after ``process_action``):
          The cache still holds TracedTensors populated by ``compute_group``.
          Action terms that read state (e.g. ``RelativeJointPositionAction``
          reading ``joint_pos``) get those TracedTensors from the cache,
          keeping the LEAPP graph connected.  After ``output_tensors()`` the
          cache is cleared so subsequent decimation sub-steps read fresh values.

        - **Non-tracing passes** (remaining decimation sub-steps and all
          subsequent iterations): The cache is cleared **before** running
          action terms so every ``.data`` read returns the current simulator
          value, preserving simulation correctness.

        Args:
            action_manager: Action manager whose instance methods should be
                wrapped.
        """
        original_process = action_manager.process_action
        original_apply = action_manager.apply_action
        task_name = self.task_name
        cache = self._annotated_tensor_cache

        def patched_process_action(action: torch.Tensor):
            """Register raw_action buffers, call real process_action, preserve action clone."""
            original_process(action)
            action_manager._action = action.clone()
            self._pending_action_output_export = True

        def patched_apply_action():
            """Coordinate cache lifecycle and LEAPP output annotation."""
            if not self._pending_action_output_export:
                cache.clear()
                return original_apply()

            # Tracing pass: cache still holds TracedTensors from compute_group.
            self._action_output_cache.clear()
            self._captured_write_term_names.clear()
            original_apply()

            self._action_output_cache.extend(self._collect_action_outputs(action_manager))
            self._action_output_cache.extend(self._collect_processed_action_fallbacks(action_manager))
            if self._uses_last_action_state:
                annotate.update_state(task_name, {"last_action": action_manager._action})
            fallback_terms = self._fallback_term_names
            static_values = self._collect_action_static_outputs(action_manager, fallback_terms)
            annotate.output_tensors(
                task_name,
                self._action_output_cache,
                static_outputs=static_values or None,
                export_with=self.export_method,
            )
            self._pending_action_output_export = False
            self._action_output_cache.clear()
            cache.clear()
            return None

        action_manager.process_action = patched_process_action
        action_manager.apply_action = patched_apply_action

    # ── Observation term wrappers ─────────────────────────────────

    @staticmethod
    def _wrap_with_proxy(original_func, proxy_env):
        """Wrap a term function so it receives the proxy env.

        Args:
            original_func: Original observation term function or manager term.
            proxy_env: Proxy environment routed into the wrapped callable.

        Returns:
            Wrapped callable that substitutes ``proxy_env`` for the real env.
        """

        if isinstance(original_func, ManagerTermBase):
            return _ManagerTermProxy(original_func, proxy_env)

        def wrapped(*args, **kwargs):
            """Invoke the original function with the proxy environment.

            Args:
                *args: Original positional arguments.
                **kwargs: Original keyword arguments.

            Returns:
                Result of the wrapped observation term.
            """
            if args:
                args = (proxy_env, *args[1:])
            else:
                args = (proxy_env,)
            return original_func(*args, **kwargs)

        wrapped.__name__ = getattr(original_func, "__name__", "unknown")
        return wrapped

    def _wrap_last_action(self, original_func):
        """Wrap ``last_action`` as a LEAPP state tensor.

        ``last_action`` is feedback state, not a regular dangling input.  We
        therefore register it through ``annotate.state_tensors(...)`` on the
        observation side and update it through ``annotate.update_state(...)``
        after the traced action pass.

        Args:
            original_func: Original ``last_action`` observation term.

        Returns:
            Wrapped callable that exports ``last_action`` as LEAPP state.
        """
        task_name = self.task_name

        def wrapped(env, action_name=None, **kwargs):
            """Run the wrapped ``last_action`` term and annotate its output.

            Args:
                env: Environment passed by the observation manager.
                action_name: Optional action term name.
                **kwargs: Additional keyword arguments for the term.

            Returns:
                Annotated last-action tensor.
            """
            result = original_func(env, action_name, **kwargs)
            return annotate.state_tensors(task_name, {"last_action": result})

        wrapped.__name__ = original_func.__name__
        return wrapped

    def _wrap_generated_commands(self, original_func, term_cfg):
        """Wrap the ``generated_commands`` observation term to annotate its output as a LEAPP input.

        Resolves command semantics (kind, element_names) from the command manager
        configuration when available.

        Args:
            original_func: Original ``generated_commands`` observation term.
            term_cfg: Observation term config used to resolve the command name.

        Returns:
            Wrapped callable that exports generated commands as LEAPP inputs.
        """
        task_name = self.task_name
        command_name_from_cfg = term_cfg.params.get("command_name")

        def wrapped(env, command_name=None, **kwargs):
            """Run the wrapped command term and annotate its output.

            Args:
                env: Environment passed by the observation manager.
                command_name: Optional command term name override.
                **kwargs: Additional keyword arguments for the term.

            Returns:
                Annotated command tensor.
            """
            result = original_func(env, command_name, **kwargs)
            leapp_input_name = command_name or command_name_from_cfg or "commands"
            command_cfg = None
            with suppress(AttributeError, KeyError):
                command_cfg = env.command_manager.get_term(leapp_input_name).cfg
            sem = TensorSemantics(
                name=leapp_input_name,
                ref=result,
                kind=getattr(command_cfg, "cmd_kind", None),
                element_names=getattr(command_cfg, "element_names", None),
                extra=build_command_connection(leapp_input_name),
            )
            return annotate.input_tensors(task_name, sem)

        wrapped.__name__ = original_func.__name__
        return wrapped

    # ── Output collection ─────────────────────────────────────────

    def _collect_action_outputs(self, action_manager) -> list[TensorSemantics]:
        """Collect non-writer action tensors that should be exported.

        Args:
            action_manager: Action manager whose terms should be inspected.

        Returns:
            Exportable tensor semantics for dynamic action outputs such as OSC
            gains.
        """
        tensors: list[TensorSemantics] = []
        for term_name, term in action_manager._terms.items():
            osc = getattr(term, "_osc", None)
            if osc and hasattr(osc, "cfg") and osc.cfg.impedance_mode in VARIABLE_IMPEDANCE_MODES:
                asset = getattr(term, "_asset", None)
                real_asset = getattr(asset, "_real_asset", asset)
                joint_ids = getattr(term, "_joint_ids", None)
                joint_names = getattr(real_asset, "joint_names", None) if real_asset else None
                scene_key = self._action_term_scene_keys.get(term_name, "ego")
                tensors.append(
                    TensorSemantics(
                        name=f"{term_name}_kp_gains",
                        ref=torch.diagonal(osc._motion_p_gains_task, dim1=-2, dim2=-1),
                        kind="kp",
                        element_names=select_element_names(joint_names, joint_ids),
                        extra=build_write_connection(scene_key, "write_joint_stiffness_to_sim_index"),
                    )
                )
                tensors.append(
                    TensorSemantics(
                        name=f"{term_name}_kd_gains",
                        ref=torch.diagonal(osc._motion_d_gains_task, dim1=-2, dim2=-1),
                        kind="kd",
                        element_names=select_element_names(joint_names, joint_ids),
                        extra=build_write_connection(scene_key, "write_joint_damping_to_sim_index"),
                    )
                )
        return tensors

    def _collect_processed_action_fallbacks(self, action_manager) -> list[TensorSemantics]:
        """Fallback: use ``term.processed_actions`` for terms that produced no write outputs.

        When an action term does not call any ``_leapp_semantics``-decorated write method
        (e.g. ``PreTrainedPolicyAction`` which delegates writes to a nested sub-policy),
        we fall back to capturing ``term.processed_actions`` as the output tensor.

        Args:
            action_manager: Action manager whose terms should be inspected.

        Returns:
            Fallback tensor semantics built from ``processed_actions``.
        """
        logger = logging.getLogger(__name__)
        fallback_terms: set[str] = set()
        tensors: list[TensorSemantics] = []
        for term_name, term in action_manager._terms.items():
            if term_name in self._captured_write_term_names:
                continue
            processed = getattr(term, "processed_actions", None)
            if processed is None:
                continue
            if isinstance(processed, torch.Tensor):
                logger.warning(
                    "Action term '%s' did not write to any asset directly. Falling back to processed_actions as the"
                    " export output.\nIf you wish to add semantic data to this policy, you need to manually annotate it"
                    " with output_tensors.",
                    term_name,
                )
                tensors.append(
                    TensorSemantics(
                        name=term_name,
                        ref=processed.clone(),
                        kind=None,
                        element_names=None,
                    )
                )
                fallback_terms.add(term_name)
        self._fallback_term_names = fallback_terms
        return tensors

    def _collect_action_static_outputs(
        self, action_manager, skip_terms: set[str] | None = None
    ) -> list[TensorSemantics]:
        """Collect static kp/kd gain values from action terms for export metadata.

        Terms in ``skip_terms`` are excluded — these are terms that fell back
        to ``processed_actions`` and whose static gains (kp/kd) belong to a
        lower abstraction level that is not part of the exported policy.

        Args:
            action_manager: Action manager whose terms should be inspected.
            skip_terms: Action term names whose static outputs should be
                skipped.

        Returns:
            Static tensor semantics for action gains exported as metadata.
        """
        static_values: list[TensorSemantics] = []
        for term_name, term in action_manager._terms.items():
            if skip_terms and term_name in skip_terms:
                continue
            osc = getattr(term, "_osc", None)
            if osc and hasattr(osc, "cfg") and osc.cfg.impedance_mode in VARIABLE_IMPEDANCE_MODES:
                continue
            asset = getattr(term, "_asset", None)
            real_asset = getattr(asset, "_real_asset", asset)
            if real_asset and hasattr(real_asset, "data"):
                data = real_asset.data
                joint_ids = getattr(term, "_joint_ids", None)
                joint_names = getattr(real_asset, "joint_names", None)
                scene_key = self._action_term_scene_keys.get(term_name, "ego")
                if hasattr(data, "default_joint_stiffness") and data.default_joint_stiffness is not None:
                    gains = data.default_joint_stiffness.torch
                    static_values.append(
                        TensorSemantics(
                            name=f"{term_name}_kp_gains",
                            ref=gains[:, joint_ids] if joint_ids else gains,
                            kind="kp",
                            element_names=select_element_names(joint_names, joint_ids),
                            extra=build_write_connection(scene_key, "write_joint_stiffness_to_sim_index"),
                        )
                    )
                if hasattr(data, "default_joint_damping") and data.default_joint_damping is not None:
                    gains = data.default_joint_damping.torch
                    static_values.append(
                        TensorSemantics(
                            name=f"{term_name}_kd_gains",
                            ref=gains[:, joint_ids] if joint_ids else gains,
                            kind="kd",
                            element_names=select_element_names(joint_names, joint_ids),
                            extra=build_write_connection(scene_key, "write_joint_damping_to_sim_index"),
                        )
                    )
        return static_values


# ══════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════


def patch_env_for_export(
    env: ManagerBasedEnv,
    export_method: str,
    required_obs_groups: set[str] | None = None,
) -> None:
    """Patch the env's observation and action managers for LEAPP export.

    This is a thin public entry point around ``ExportPatcher``.  It mutates
    the provided env instance in-place so that:

    - Observation terms route through proxy objects that annotate tensor
      reads from **any** scene entity data class (articulations, rigid
      objects, sensors, etc.).
    - Action terms route through proxy objects that annotate both data
      reads **and** ``Articulation`` write methods.

    Data properties are resolved lazily through proxies — no hardcoded
    class list is required. To produce LEAPP input annotations, the
    accessed data property getter must carry ``_leapp_semantics``.
    Likewise, action-side write methods must be annotated to produce
    semantic LEAPP outputs. Undecorated reads and writes are forwarded
    as normal runtime access, but they do not gain semantic annotation
    metadata through this patching path.

    State reads are deduplicated across observation and action paths via a
    shared cache, so a property like ``joint_pos`` that is read by both an
    observation term and a relative-position action term appears as a single
    LEAPP input edge.

    The underlying env, scene, assets, and tensors remain shared with the rest
    of the pipeline; only the manager call paths are redirected.

    Args:
        env: Manager-based environment to patch in place.
        export_method: LEAPP export backend passed to
            :func:`annotate.output_tensors`.
        required_obs_groups: Observation groups that should be patched, or
            ``None`` to patch all groups.
    """
    patcher = ExportPatcher(export_method, required_obs_groups=required_obs_groups)
    patcher.setup(env)
