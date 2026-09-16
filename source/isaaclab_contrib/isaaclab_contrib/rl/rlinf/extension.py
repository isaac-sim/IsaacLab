# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RLinf extension module for IsaacLab tasks.

This module is loaded by RLinf's Worker._load_user_extensions() when
RLINF_EXT_MODULE=isaaclab_contrib.rl.rlinf.extension is set in the environment.

It registers IsaacLab tasks into RLinf's registries, allowing IsaacLab users
to train on their tasks without modifying RLinf source code.

Configuration is read from the Hydra YAML config under `env.train.isaaclab`:
    env:
      train:
        isaaclab: &isaaclab_config  # YAML anchor for reuse
          task_description: "..."
          main_images: "front_camera"
          extra_view_images: ["left_wrist_camera", "right_wrist_camera"]
          states:
            - key: "robot_joint_state"
              slice: [15, 29]
          gr00t_mapping:
            video:
              main_images: "video.room_view"
              ...
          action_mapping:
            prefix_pad: 15
      eval:
        isaaclab: *isaaclab_config  # Reuse via YAML anchor

Task IDs are read automatically from ``env.train.init_params.id`` and
``env.eval.init_params.id`` in the YAML config.

Usage:
    export RLINF_EXT_MODULE=isaaclab_contrib.rl.rlinf.extension
    export RLINF_CONFIG_FILE=/path/to/isaaclab_ppo_gr00t_assemble_trocar.yaml
"""

from __future__ import annotations

import collections.abc
import importlib
import logging
import os
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import yaml
from rlinf.models.embodiment.gr00t import embodiment_tags

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

_registered = False

# Cache for YAML config (loaded once per process)
_full_cfg_cache: dict | None = None


def register() -> None:
    """Register IsaacLab extensions into RLinf.

    This function is called automatically by RLinf's Worker._load_user_extensions()
    when RLINF_EXT_MODULE=isaaclab_contrib.rl.rlinf.extension is set.

    It performs the following registrations:
    1. Registers GR00T obs/action converters
    2. Registers GR00T data config
    3. Patches GR00T get_model for custom embodiment
    4. Registers task IDs from YAML config (env.*.init_params.id) into REGISTER_ISAACLAB_ENVS
    """
    global _registered
    if _registered:
        return
    _registered = True

    logger.info("isaaclab_contrib.rl.rlinf.extension: Registering IsaacLab extensions...")

    # Load config once and pass to all registration functions
    cfg = _get_isaaclab_cfg()

    _register_gr00t_converters(cfg)
    _patch_gr00t_get_model(cfg)
    _register_isaaclab_envs()

    logger.info("isaaclab_contrib.rl.rlinf.extension: Registration complete.")


def _load_full_cfg() -> dict:
    """Load and cache the full YAML config from ``RLINF_CONFIG_FILE``.

    Raises:
        ValueError: If the ``RLINF_CONFIG_FILE`` environment variable is not set.

    Returns:
        The parsed YAML config as a nested dictionary.
    """
    global _full_cfg_cache
    if _full_cfg_cache is not None:
        return _full_cfg_cache
    config_file = os.environ.get("RLINF_CONFIG_FILE", "")
    if not config_file:
        raise ValueError("RLINF_CONFIG_FILE not set")
    with open(config_file) as f:
        _full_cfg_cache = yaml.safe_load(f)
    logger.info(f"Loaded full config from {config_file}")
    return _full_cfg_cache


def _gr00t_model_type() -> str:
    """Return ``actor.model.model_type`` from the cached config (``gr00t`` when unset).

    RLinf dispatches on this value; ``gr00t_n1d7`` selects the GR00T N1.7 code paths.
    """
    return str(_load_full_cfg().get("actor", {}).get("model", {}).get("model_type", "gr00t"))


def _get_isaaclab_cfg() -> dict:
    """Return the ``env.train.isaaclab`` section from the cached full config.

    Returns:
        The IsaacLab-specific configuration dictionary. Empty dict if the section is missing.
    """
    return _load_full_cfg().get("env", {}).get("train", {}).get("isaaclab", {})


def _patch_embodiment_tags(cfg: dict) -> None:
    """Add custom embodiment tag to RLinf's EmbodimentTag enum and mapping if needed.

    Reads ``embodiment_tag`` and ``embodiment_tag_id`` from the IsaacLab config section.
    Only adds the tag if it is not already present in RLinf's native registry.

    Args:
        cfg: The IsaacLab-specific configuration dictionary (``env.train.isaaclab``).
    """
    # GR00T uses embodiment tags to identify different robots.  Custom robots
    # (like G129+Dex3) need a unique tag string and numeric ID so that the
    # model's tokenizer can map them to the correct action/state dimensions.
    #
    # The numeric ID is the projector index in GR00T's Action Expert Module.
    # Known mapping (from gr00t/data/embodiment_tags.py):
    #   17 = oxe_droid, 24 = gr1, 26 = agibot_genie1, 31 = new_embodiment
    # Default 31 corresponds to the "new_embodiment" slot reserved for
    # fine-tuning on custom robots.
    embodiment_tag = cfg.get("embodiment_tag", "new_embodiment")
    tag_id = cfg.get("embodiment_tag_id", 31)

    # If tag is already in registry (native or previously added), skip
    if embodiment_tag in embodiment_tags.EMBODIMENT_TAG_MAPPING:
        logger.info(f"embodiment_tag '{embodiment_tag}' already registered")
        return
    # Add to enum
    tag_upper = embodiment_tag.upper().replace("-", "_")
    if not hasattr(embodiment_tags.EmbodimentTag, tag_upper):
        existing_members = {e.name: e.value for e in embodiment_tags.EmbodimentTag}
        existing_members[tag_upper] = embodiment_tag
        NewEmbodimentTag = Enum("EmbodimentTag", existing_members)

        embodiment_tags.EmbodimentTag = NewEmbodimentTag
        logger.info(f"Added EmbodimentTag.{tag_upper} = '{embodiment_tag}'")

    # Add to mapping
    embodiment_tags.EMBODIMENT_TAG_MAPPING[embodiment_tag] = tag_id
    logger.info(f"Added EMBODIMENT_TAG_MAPPING['{embodiment_tag}'] = {tag_id}")


def _patch_gr00t_get_model(cfg: dict) -> None:
    """Monkeypatch RLinf's GR00T ``get_model`` to support custom ``data_config``.

    The patch is applied only when the user specifies a ``data_config_class`` in the
    YAML config. Embodiment tags are always ensured to be registered.

    Args:
        cfg: The IsaacLab-specific configuration dictionary (``env.train.isaaclab``).
    """
    # Always ensure embodiment tag is registered
    _patch_embodiment_tags(cfg)
    # Only patch get_model if user wants custom data_config
    data_config_class = cfg.get("data_config_class", "")
    if not data_config_class:
        logger.info("No data_config_class specified, using RLinf's default get_model")
        return

    if _gr00t_model_type() == "gr00t_n1d7":
        logger.info("model_type=gr00t_n1d7: RLinf's native N1.7 loader reads the checkpoint processor")
        return

    import rlinf.models.embodiment.gr00t as rlinf_gr00t_mod

    def patched_get_model(model_cfg, torch_dtype=None) -> object:
        """Load a GR00T model with custom ``data_config`` and embodiment tag.

        Args:
            model_cfg: RLinf model configuration object containing ``model_path``,
                ``embodiment_tag``, ``denoising_steps``, ``num_action_chunks``,
                ``obs_converter_type``, and ``rl_head_config``.
            torch_dtype: The torch dtype for the model. Defaults to ``torch.bfloat16``.

        Raises:
            FileNotFoundError: If ``model_cfg.model_path`` does not exist.

        Returns:
            The loaded GR00T model instance.
        """
        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        # Handle custom embodiment (we only get here if tag was not natively supported)
        from gr00t.experiment.data_config import load_data_config

        try:
            from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import GR00T_N1_5_ForRLActionPrediction
        except ImportError:  # RLinf < 0.3 kept the N1.5 model at the package root
            from rlinf.models.embodiment.gr00t.gr00t_action_model import GR00T_N1_5_ForRLActionPrediction
        from rlinf.models.embodiment.gr00t.utils import replace_dropout_with_identity
        from rlinf.utils.patcher import Patcher

        # Apply RLinf's standard EmbodimentTag patches
        Patcher.clear()
        Patcher.add_patch(
            "gr00t.data.embodiment_tags.EmbodimentTag",
            "rlinf.models.embodiment.gr00t.embodiment_tags.EmbodimentTag",
        )
        Patcher.add_patch(
            "gr00t.data.embodiment_tags.EMBODIMENT_TAG_MAPPING",
            "rlinf.models.embodiment.gr00t.embodiment_tags.EMBODIMENT_TAG_MAPPING",
        )
        Patcher.apply()

        data_config = load_data_config(data_config_class)
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        model_path = Path(model_cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # rl_model_path: optional path to an RLinf checkpoint with full_weights.pt
        rl_model_path = getattr(model_cfg, "rl_model_path", None)

        model = GR00T_N1_5_ForRLActionPrediction.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            embodiment_tag=model_cfg.embodiment_tag,
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=model_cfg.denoising_steps,
            output_action_chunks=model_cfg.num_action_chunks,
            obs_converter_type=model_cfg.obs_converter_type,
            tune_visual=False,
            tune_llm=False,
            rl_head_config=model_cfg.rl_head_config,
        )

        if rl_model_path:
            # Accept either the full_weights.pt file itself or the global_step_<N> directory holding it.
            rl_weights = Path(rl_model_path)
            if rl_weights.is_dir():
                rl_weights = rl_weights / "actor" / "model_state_dict" / "full_weights.pt"
            if not rl_weights.exists():
                raise FileNotFoundError(
                    f"rl_model_path={rl_model_path}: cannot find full_weights.pt "
                    f"(tried directly and under actor/model_state_dict/)"
                )
            logger.info(f"Loading RL finetuned weights from {rl_weights}")
            state_dict = torch.load(rl_weights, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)

        model.to(torch_dtype)
        if model_cfg.rl_head_config.add_value_head:
            model.action_head.value_head._init_weights()
        if model_cfg.rl_head_config.disable_dropout:
            replace_dropout_with_identity(model)

        logger.info(f"Loaded GR00T model with embodiment_tag='{model_cfg.embodiment_tag}'")
        return model

    rlinf_gr00t_mod.get_model = patched_get_model
    logger.info(f"Patched get_model for data_config_class='{data_config_class}'")


def _resolve_action_converter(cfg: dict):
    """Return the GR00T -> IsaacLab action converter for this task.

    GR00T N1.7 checkpoints take no ``data_config_class``; a task instead names a
    ``modality_config_module`` whose import registers the embodiment's modality layout with GR00T.
    When that module also defines ``convert_gr00t_to_isaaclab_action``, it replaces the generic
    prefix/suffix-padding converter. The module is only imported for ``gr00t_n1d7`` because it
    depends on the N1.7 release of ``gr00t``.

    Args:
        cfg: The IsaacLab-specific configuration dictionary (``env.train.isaaclab``).
    """
    module_name = cfg.get("modality_config_module", "")
    if not module_name or _gr00t_model_type() != "gr00t_n1d7":
        return _convert_gr00t_to_isaaclab_action
    module = importlib.import_module(module_name)
    logger.info(f"Imported GR00T N1.7 modality config module: {module_name}")
    return getattr(module, "convert_gr00t_to_isaaclab_action", _convert_gr00t_to_isaaclab_action)


def _resolve_state_to_action_indices(cfg: dict) -> list[int] | None:
    """Return where the policy's joint-state vector lands in the action vector, if the task says so.

    Needed to command "hold the current pose". A task whose action order differs from its policy
    order — H2 interleaves the two hands, so the state is not a contiguous slice of the action —
    publishes the mapping as ``POLICY_STATE_TO_ACTION_INDICES`` in its ``modality_config_module``.

    Args:
        cfg: The IsaacLab-specific configuration dictionary (``env.train.isaaclab``).

    Returns:
        Action-vector index for each state entry, or ``None`` when the task publishes no mapping.
    """
    module_name = cfg.get("modality_config_module", "")
    if not module_name or _gr00t_model_type() != "gr00t_n1d7":
        return None
    indices = getattr(importlib.import_module(module_name), "POLICY_STATE_TO_ACTION_INDICES", None)
    return None if indices is None else list(indices)


def _extract_states(policy_obs: dict, cfg: dict) -> torch.Tensor | None:
    """Concatenate the state terms a task lists under ``states`` into one vector.

    Args:
        policy_obs: The environment's policy observation group.
        cfg: The IsaacLab-specific configuration dictionary (``env.train.isaaclab``).

    Returns:
        The concatenated state of shape ``(B, D)``, or ``None`` when no listed term is present.
    """
    state_parts = []
    for spec in cfg.get("states") or []:
        if isinstance(spec, str):
            state = policy_obs.get(spec)
        else:
            state = policy_obs.get(spec.get("key"))
            slice_range = spec.get("slice") if state is not None else None
            if slice_range:
                state = state[:, slice_range[0] : slice_range[1]]
        if state is not None:
            state_parts.append(state)
    return torch.cat(state_parts, dim=-1) if state_parts else None


def _apply_reset_sensor_refresh(env) -> None:
    """Step once after every reset so the first observation shows the new episode.

    ``TiledCamera`` refreshes its annotators only through a real simulation step: the observation
    returned by ``reset`` still carries the previous episode's image, which the policy would then
    act on. Re-commanding the robot's current joint positions advances the render without moving
    it. Requires the task to publish :func:`_resolve_state_to_action_indices`; other tasks are
    left untouched.

    Args:
        env: The unwrapped IsaacLab environment to patch in place.
    """
    cfg = _get_isaaclab_cfg()
    indices = _resolve_state_to_action_indices(cfg)
    if indices is None:
        return

    action_indices = torch.as_tensor(indices, dtype=torch.long, device=env.device)
    original_reset = env.reset

    def reset_then_refresh(*args, **kwargs):
        obs, extras = original_reset(*args, **kwargs)
        states = _extract_states(obs.get("policy", obs), cfg)
        if states is None:
            return obs, extras
        # Zero raw actions map to the default pose for the joints the policy does not predict
        # (see the action term's offset), so this holds the whole robot where the reset put it.
        hold = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        hold[:, action_indices] = states.to(device=hold.device, dtype=hold.dtype)
        obs, _, _, _, extras = env.step(hold)
        return obs, extras

    env.reset = reset_then_refresh


def _register_gr00t_converters(cfg: dict) -> None:
    """Register GR00T obs/action converters for IsaacLab tasks.

    Reads ``obs_converter_type`` from the YAML config (``env.train.isaaclab.obs_converter_type``)
    and registers the corresponding observation and action conversion functions into
    RLinf's ``simulation_io`` registry.

    Args:
        cfg: The IsaacLab-specific configuration dictionary (``env.train.isaaclab``).
    """
    from rlinf.models.embodiment.gr00t import simulation_io

    obs_converter_type = cfg.get("obs_converter_type", "dex3")

    if obs_converter_type not in simulation_io.OBS_CONVERSION:
        simulation_io.OBS_CONVERSION[obs_converter_type] = _convert_isaaclab_obs_to_gr00t
        logger.info(f"Registered obs converter: {obs_converter_type}")

    action_converter = _resolve_action_converter(cfg)
    # RLinf < 0.3 keeps one ACTION_CONVERSION table; newer versions split it per GR00T generation.
    for registry_name in ("ACTION_CONVERSION", "ACTION_CONVERSION_N1D5", "ACTION_CONVERSION_N1D7"):
        registry = getattr(simulation_io, registry_name, None)
        if isinstance(registry, dict) and obs_converter_type not in registry:
            registry[obs_converter_type] = action_converter
            logger.info(f"Registered action converter into {registry_name}: {obs_converter_type}")


def _convert_isaaclab_obs_to_gr00t(env_obs: dict) -> dict:
    """Convert IsaacLab env observations to GR00T format.

    Uses ``gr00t_mapping`` from the YAML config (``env.train.isaaclab.gr00t_mapping``)
    to map IsaacLab observation keys to GR00T-expected keys.

    Args:
        env_obs: Observation dictionary from ``_wrap_obs`` with the following keys:

            - ``"main_images"``: ``(B, H, W, C)`` torch tensor.
            - ``"extra_view_images"``: ``(B, N, H, W, C)`` torch tensor.
            - ``"states"``: ``(B, D)`` torch tensor.
            - ``"task_descriptions"``: list of strings.

    Returns:
        A dictionary with GR00T-formatted observations (numpy arrays with a time
        dimension, e.g. ``(B, T=1, H, W, C)``).
    """
    groot_obs = {}
    # Load mapping config from YAML or env var
    cfg = _get_isaaclab_cfg()
    gr00t_mapping = cfg.get("gr00t_mapping", {})
    video_mapping = gr00t_mapping.get("video", {})
    state_mapping = gr00t_mapping.get("state", [])
    # Convert main_images -> video.xxx
    if "main_images" in env_obs:
        main = env_obs["main_images"]
        gr00t_key = video_mapping.get("main_images", "video.room_view")
        if isinstance(main, torch.Tensor):
            # (B, H, W, C) -> (B, T=1, H, W, C)
            groot_obs[gr00t_key] = main.unsqueeze(1).cpu().numpy()
    # Convert extra_view_images -> video.xxx
    if "extra_view_images" in env_obs:
        extra = env_obs["extra_view_images"]  # (B, N, H, W, C)
        extra_keys = video_mapping.get("extra_view_images", [])
        if isinstance(extra, torch.Tensor):
            for i, key in enumerate(extra_keys):
                if i < extra.shape[1]:
                    # (B, H, W, C) -> (B, T=1, H, W, C)
                    groot_obs[key] = extra[:, i].unsqueeze(1).cpu().numpy()
    # Convert states -> state.xxx with slicing
    if "states" in env_obs and state_mapping:
        states = env_obs["states"]  # (B, D)
        if isinstance(states, torch.Tensor):
            states_np = states.unsqueeze(1).cpu().numpy()  # (B, T=1, D)
            for spec in state_mapping:
                gr00t_key = spec.get("gr00t_key")
                slice_range = spec.get("slice", [0, states_np.shape[-1]])
                if gr00t_key:
                    groot_obs[gr00t_key] = states_np[:, :, slice_range[0] : slice_range[1]]

    # Pass through task descriptions
    groot_obs["annotation.human.action.task_description"] = env_obs.get("task_descriptions", [])
    # N1.7's processor reads the non-action annotation key, while N1.5's transform asserts that exactly
    # one ``annotation.*`` key is present, so the extra key is only added for N1.7.
    if _gr00t_model_type() == "gr00t_n1d7":
        groot_obs["annotation.human.task_description"] = groot_obs["annotation.human.action.task_description"]

    return groot_obs


def _convert_gr00t_to_isaaclab_action(action_chunk: dict, chunk_size: int = 1) -> np.ndarray:
    """Convert GR00T action output to IsaacLab env action format.

    Uses ``action_mapping`` from the YAML config (``env.train.isaaclab.action_mapping``)
    to apply optional prefix/suffix zero-padding to the concatenated action vector.

    Args:
        action_chunk: Dictionary of action arrays from GR00T, each with shape
            ``(B, T, D_i)``.
        chunk_size: Number of time steps to keep from the action chunk. Defaults to 1.

    Returns:
        Concatenated and padded action array with shape ``(B, chunk_size, D)``.
    """

    # Load mapping config from YAML or env var
    cfg = _get_isaaclab_cfg()
    action_mapping = cfg.get("action_mapping", {})
    prefix_pad = action_mapping.get("prefix_pad", 0)
    suffix_pad = action_mapping.get("suffix_pad", 0)

    # Concatenate all action parts
    action_parts = [v[:, :chunk_size, :] for v in action_chunk.values()]
    action_concat = np.concatenate(action_parts, axis=-1)

    # Apply padding
    if prefix_pad > 0 or suffix_pad > 0:
        action_concat = np.pad(
            action_concat,
            ((0, 0), (0, 0), (prefix_pad, suffix_pad)),
            mode="constant",
            constant_values=0,
        )
    return action_concat


def _register_isaaclab_envs() -> None:
    """Register IsaacLab tasks into RLinf's REGISTER_ISAACLAB_ENVS map.

    Task IDs are read from ``env.train.init_params.id`` and
    ``env.eval.init_params.id`` in the YAML config.
    """
    from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS

    # Collect unique task IDs from the YAML config (train + eval)
    full_cfg = _load_full_cfg()
    env_cfg = full_cfg.get("env", {})
    task_ids: list[str] = []
    for section in ("train", "eval"):
        tid = env_cfg.get(section, {}).get("init_params", {}).get("id", "")
        if tid and tid not in task_ids:
            task_ids.append(tid)

    if not task_ids:
        logger.warning("No task IDs found in YAML config (env.*.init_params.id)")
        return

    logger.info(f"Tasks to register: {task_ids}")

    for task_id in task_ids:
        if task_id in REGISTER_ISAACLAB_ENVS:
            logger.debug(f"Task '{task_id}' already registered, skipping")
            continue

        # Create a generic wrapper class for this task
        env_class = _create_generic_env_wrapper(task_id)
        REGISTER_ISAACLAB_ENVS[task_id] = env_class
        logger.info(f"Registered IsaacLab task '{task_id}' for RLinf")

    logger.debug(f"REGISTER_ISAACLAB_ENVS now contains: {list(REGISTER_ISAACLAB_ENVS.keys())}")


def _create_generic_env_wrapper(task_id: str) -> type:
    """Create a generic wrapper class for an IsaacLab task.

    The wrapper class will load the task configuration at runtime
    (after AppLauncher starts) and configure observation mapping accordingly.

    This follows the same pattern as i4h's rlinf_ext: all isaaclab-dependent
    imports happen inside _make_env_function, after AppLauncher starts.

    Args:
        task_id: The gymnasium task ID.

    Returns:
        A class that inherits from IsaaclabBaseEnv.
    """
    from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv

    _task_id = task_id

    class IsaacLabGenericEnv(IsaaclabBaseEnv):
        """Generic environment wrapper for IsaacLab tasks.

        Config is read from the YAML file via ``_get_isaaclab_cfg()``.
        """

        def __init__(self, cfg, num_envs: int, seed_offset: int, total_num_processes: int, worker_info):
            """Initialize the generic IsaacLab environment wrapper.

            Args:
                cfg: RLinf environment configuration object.
                num_envs: Number of parallel environments.
                seed_offset: Seed offset for reproducibility.
                total_num_processes: Total number of worker processes.
                worker_info: RLinf worker metadata.
            """
            super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

        def _record_metrics(self, step_reward, terminations, infos):
            """Override to use terminations (task completion) for success_once."""

            episode_info = {}
            self.returns += step_reward
            self.success_once = self.success_once | terminations.bool()
            episode_info["success_once"] = self.success_once.clone()
            episode_info["return"] = self.returns.clone()
            episode_info["episode_len"] = self.elapsed_steps.clone()
            episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
            infos["episode"] = episode_info
            return infos

        def chunk_step(self, chunk_actions):
            """Execute an action chunk, holding the pose of sub-environments that reset mid-chunk.

            ``ManagerBasedRLEnv.step`` resets a terminated sub-environment before returning, so the
            remainder of the chunk — predicted from the episode that just ended — would otherwise be
            applied to the fresh one, yanking the robot toward the previous episode's final pose.
            When the task sets ``hold_pose_on_midchunk_reset``, those actions are replaced by the
            post-reset joint positions, leaving the robot still until the next chunk boundary
            produces policy output for the new episode.

            The masking is installed on ``self.step`` for the duration of the chunk rather than by
            reimplementing the chunk loop, so RLinf keeps doing its own reward and done bookkeeping.

            Args:
                chunk_actions: Actions of shape ``(num_envs, chunk_size, action_dim)``.

            Returns:
                The tuple returned by ``IsaaclabBaseEnv.chunk_step``.
            """
            cfg = _get_isaaclab_cfg()
            if not cfg.get("hold_pose_on_midchunk_reset", False):
                return super().chunk_step(chunk_actions)

            indices = _resolve_state_to_action_indices(cfg)
            action_indices = None if indices is None else torch.as_tensor(indices, dtype=torch.long, device=self.device)
            done_so_far = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            last_states = [None]
            base_step = super().step

            def step_holding_reset_envs(actions, auto_reset=True):
                states = last_states[0]
                if states is not None and bool(done_so_far.any()):
                    rows = done_so_far.nonzero(as_tuple=False).squeeze(-1)
                    hold = states[rows].to(device=actions.device, dtype=actions.dtype)
                    actions = actions.clone()
                    if action_indices is None:
                        # Without a published mapping the state is assumed to occupy the action's
                        # tail, which is how ``action_mapping.prefix_pad`` lays it out.
                        actions[rows, actions.shape[-1] - hold.shape[-1] :] = hold
                    else:
                        # Zero raw actions hold the joints the policy does not predict at their
                        # default pose (see the action term's offset).
                        actions[rows] = 0.0
                        actions[rows[:, None], action_indices[None, :]] = hold
                obs, step_reward, terminations, truncations, infos = base_step(actions, auto_reset=auto_reset)
                last_states[0] = obs.get("states")
                done_so_far.logical_or_((terminations | truncations).bool())
                return obs, step_reward, terminations, truncations, infos

            self.step = step_holding_reset_envs
            try:
                return super().chunk_step(chunk_actions)
            finally:
                del self.step

        def _make_env_function(self) -> collections.abc.Callable:
            """Create the environment factory function.

            This function runs in a child process (via ``SubProcIsaacLabEnv``).
            All IsaacLab-dependent imports happen here, after ``AppLauncher`` starts.

            Returns:
                A callable that creates and returns the IsaacLab environment and sim app.
            """

            def make_env_isaaclab() -> tuple:
                """Create the IsaacLab environment inside the child process.

                Returns:
                    A tuple of ``(env, sim_app)`` where ``env`` is the unwrapped
                    gymnasium environment and ``sim_app`` is the Isaac Sim application.
                """
                from isaaclab.app import AppLauncher

                sim_app = AppLauncher(headless=True, enable_cameras=True).app
                import gymnasium as gym

                from isaaclab_tasks.utils import parse_env_cfg

                isaac_env_cfg = parse_env_cfg(self.isaaclab_env_id, num_envs=self.cfg.init_params.num_envs)

                env = gym.make(self.isaaclab_env_id, cfg=isaac_env_cfg, render_mode="rgb_array").unwrapped
                _apply_reset_sensor_refresh(env)

                return env, sim_app

            return make_env_isaaclab

        def _wrap_obs(self, obs: dict) -> dict:
            """Convert IsaacLab observations to the RLinf format.

            The output format matches i4h's convention:

            - ``"main_images"``: ``(B, H, W, C)`` — single main camera.
            - ``"extra_view_images"``: ``(B, N, H, W, C)`` — stacked extra cameras.
            - ``"states"``: ``(B, D)`` — concatenated state vector.
            - ``"task_descriptions"``: ``list[str]`` — task descriptions.
            Config is read from the YAML file via :func:`_get_isaaclab_cfg`.

            Args:
                obs: Raw observation dictionary from the IsaacLab environment.

            Returns:
                A dictionary with observations mapped to the RLinf convention.
            """
            # import torch

            policy_obs = obs.get("policy", obs)
            camera_obs = obs.get("camera_images", {})

            cfg = _get_isaaclab_cfg()
            # Get task description from config
            task_desc = cfg.get("task_description", "") or self.task_description
            rlinf_obs = {
                "task_descriptions": [task_desc] * self.num_envs,
            }

            if not cfg:
                logger.warning("IsaacLab config is empty, returning minimal observation")
                return rlinf_obs

            # main_images: single camera key -> (B, H, W, C)
            main_key = cfg.get("main_images")
            if main_key and main_key in camera_obs:
                rlinf_obs["main_images"] = camera_obs[main_key]

            # extra_view_images: camera key(s) -> stack to (B, N, H, W, C)
            extra_keys = cfg.get("extra_view_images")
            if extra_keys:
                if isinstance(extra_keys, str):
                    extra_keys = [extra_keys]
                extra_imgs = [camera_obs[k] for k in extra_keys if k in camera_obs]
                if extra_imgs:
                    rlinf_obs["extra_view_images"] = torch.stack(extra_imgs, dim=1)

            # states: list of state specs -> concatenate to (B, D)
            states = _extract_states(policy_obs, cfg)
            if states is not None:
                rlinf_obs["states"] = states

            return rlinf_obs

        def add_image(self, obs: dict) -> np.ndarray | None:
            """Get image for video logging.

            Args:
                obs: Raw observation dictionary from the IsaacLab environment.

            Returns:
                A numpy array of shape ``(H, W, C)`` for the first environment, or
                ``None`` if no camera image is available.
            """
            camera_obs = obs.get("camera_images", {})
            cfg = _get_isaaclab_cfg()
            # Try main_images key, fallback to first available camera
            main_key = cfg.get("main_images")
            if main_key and main_key in camera_obs:
                return camera_obs[main_key][0].cpu().numpy()
            for img in camera_obs.values():
                return img[0].cpu().numpy()
            return None

    return IsaacLabGenericEnv
