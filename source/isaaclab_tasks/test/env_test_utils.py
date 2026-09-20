# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared test utilities for Isaac Lab environments."""

import os
from collections.abc import Collection

import gymnasium as gym
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.envs.utils.spaces import sample_space
from isaaclab.physics import PhysicsCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.version import get_isaac_sim_version

from isaaclab_tasks.utils.hydra import collect_presets, resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

# Map of task IDs to the reason for marking the corresponding parametrized
# test cases as expected failures.  Tests that consume :func:`setup_environment`
# automatically pick up these marks via :class:`pytest.param`.
XFAIL_TASKS: dict[str, str] = {}

# Native crashes cannot be contained by xfail because the process exits before
# pytest records an outcome. Temporarily skip these tasks in every environment smoke suite.
SKIP_TASKS: dict[str, str] = {
    "Isaac-Lift-Soft-Franka": "Temporarily skipped because the soft-lift environment can crash the test process.",
    "Isaac-Lift-Soft-Franka-Camera": (
        "Temporarily skipped because the soft-lift camera environment can crash the test process."
    ),
}

SINGLE_ENVIRONMENT_TASKS = (
    "Isaac-Cartpole",
    "Isaac-Reach-Franka",
    "Isaac-Reorient-Cube-Shadow",
    "Isaac-Velocity-Rough-AnymalD",
)


def _task_tier(task_spec) -> str | None:
    """Return ``"core"`` or ``"contrib"`` based on the task's env-config entry-point module.

    Core tasks register their config under :mod:`isaaclab_tasks.core` and contributed
    tasks under :mod:`isaaclab_tasks.contrib`. Returns ``None`` if the tier cannot be
    determined from the registered entry point.
    """
    entry = task_spec.kwargs.get("env_cfg_entry_point")
    if isinstance(entry, str):
        module = entry.split(":")[0]
        if module.startswith("isaaclab_tasks.core"):
            return "core"
        if module.startswith("isaaclab_tasks.contrib"):
            return "contrib"
    return None


def setup_environment(
    multi_agent: bool | None = None,
    physics_preset_name: str | None = None,
    tier: str | None = None,
    exclude_task_names: Collection[str] = (),
) -> list[str]:
    """
    Acquire all registered Isaac environment task IDs with optional filters.

    Args:
        multi_agent:
            - True: include only multi-agent environments
            - False: include only single-agent environments
            - None: include all environments regardless of agent type
        physics_preset_name: Include only environments that explicitly provide this physics preset.
        tier:
            - "core": include only core environments (registered under ``isaaclab_tasks.core``).
            - "contrib": include only contributed environments (registered under ``isaaclab_tasks.contrib``).
            - None: include all environments regardless of tier.
        exclude_task_names: Registered task IDs to omit from the result.

    Returns:
        A sorted list of task IDs matching the selected filters.
    """
    # disable interactive mode for wandb for automate environments
    os.environ["WANDB_DISABLED"] = "true"

    # acquire all Isaac environment names
    registered_tasks = []
    for task_spec in gym.registry.values():
        # only consider Isaac environments
        if "Isaac" not in task_spec.id:
            continue

        # apply core/contrib tier filter
        if tier is not None and _task_tier(task_spec) != tier:
            continue

        if task_spec.id in exclude_task_names:
            continue

        # apply multi agent filter
        if multi_agent is not None:
            # parse config
            env_cfg = parse_env_cfg(task_spec.id)
            if (multi_agent is True and not hasattr(env_cfg, "possible_agents")) or (
                multi_agent is False and hasattr(env_cfg, "possible_agents")
            ):
                continue
        # if None: no filter

        if physics_preset_name is not None:
            raw_cfg = load_cfg_from_registry(task_spec.id, "env_cfg_entry_point")
            physics_preset_groups = collect_presets(raw_cfg).values()
            if not any(
                physics_preset_name in preset_group and isinstance(preset_group[physics_preset_name], PhysicsCfg)
                for preset_group in physics_preset_groups
            ):
                continue

        registered_tasks.append(task_spec.id)

    # sort environments alphabetically
    registered_tasks.sort()

    # this flag is necessary to prevent a bug where the simulation gets stuck randomly when running many environments
    get_settings_manager().set_bool("/physics/cooking/ujitsoCollisionCooking", False)

    print(">>> All registered environments:", registered_tasks)

    # Apply skip before xfail so native-crash exclusions never execute.
    marked_tasks = []
    for task_id in registered_tasks:
        if task_id in SKIP_TASKS:
            marked_tasks.append(pytest.param(task_id, marks=pytest.mark.skip(reason=SKIP_TASKS[task_id])))
        elif task_id in XFAIL_TASKS:
            marked_tasks.append(
                pytest.param(task_id, marks=pytest.mark.xfail(reason=XFAIL_TASKS[task_id], strict=False))
            )
        else:
            marked_tasks.append(task_id)
    return marked_tasks


def _fire_all_interval_events_once(env) -> None:
    """Force every interval-mode event term to fire once.

    Invokes :meth:`~isaaclab.managers.EventManager.apply` with ``mode="interval"``
    and a ``dt`` larger than any plausible ``interval_range_s`` upper bound, so the
    trigger condition trips for every term in a single call. The manager re-samples
    ``time_left`` from each term's original ``interval_range_s`` after firing, so
    subsequent ``env.step()`` calls observe original interval timing.

    No-op for envs without an :class:`~isaaclab.managers.EventManager` or
    without any ``interval``-mode terms.

    Args:
        env: A constructed env instance.
    """
    event_manager = getattr(env.unwrapped, "event_manager", None)
    if event_manager is None:
        return
    if "interval" not in event_manager.available_modes:
        return
    # Pass a very large dt for (time_left -= dt) to be less than 1e-6
    event_manager.apply("interval", dt=1e9)


def _configure_osc_smoke_actions(env, actions: torch.Tensor) -> None:
    """Replace absolute OSC targets with matching task commands.

    Random samples from an unbounded action space are not valid absolute poses: their
    quaternions are not normalized and their positions may be unreachable. When an
    operational-space action term has a unique pose command for the same asset and
    body, this helper tracks that command with the controller's nominal gains.

    Args:
        env: A constructed manager-based environment.
        actions: The sampled actions to update in place.
    """
    action_manager = getattr(env.unwrapped, "action_manager", None)
    command_manager = getattr(env.unwrapped, "command_manager", None)
    if action_manager is None or command_manager is None:
        return

    action_offset = 0
    for term_name, term_dim in zip(action_manager.active_terms, action_manager.action_term_dim):
        action_term = action_manager.get_term(term_name)
        action_cfg = action_term.cfg
        if not isinstance(action_cfg, OperationalSpaceControllerActionCfg):
            action_offset += term_dim
            continue

        controller_cfg = action_cfg.controller_cfg
        if "pose_abs" not in controller_cfg.target_types or action_cfg.task_frame_rel_path is not None:
            action_offset += term_dim
            continue

        pose_commands = []
        for command_name in command_manager.active_terms:
            command_term = command_manager.get_term(command_name)
            if (
                getattr(command_term.cfg, "asset_name", None) == action_cfg.asset_name
                and getattr(command_term.cfg, "body_name", None) == action_cfg.body_name
                and command_term.command.shape[1] == 7
            ):
                pose_commands.append(command_term.command)
        if len(pose_commands) != 1:
            action_offset += term_dim
            continue

        command_offset = 0
        for target_type in controller_cfg.target_types:
            if target_type == "pose_abs":
                pose_command = pose_commands[0]
                actions[:, action_offset + command_offset : action_offset + command_offset + 3] = (
                    pose_command[:, :3] / action_cfg.position_scale
                )
                actions[:, action_offset + command_offset + 3 : action_offset + command_offset + 7] = (
                    pose_command[:, 3:] / action_cfg.orientation_scale
                )
                command_offset += 7
            elif target_type in ("pose_rel", "wrench_abs"):
                command_offset += 6

        if controller_cfg.impedance_mode in ("variable_kp", "variable"):
            stiffness = torch.as_tensor(controller_cfg.motion_stiffness_task, device=actions.device)
            actions[:, action_offset + command_offset : action_offset + command_offset + 6] = (
                stiffness / action_cfg.stiffness_scale
            )
            command_offset += 6
        if controller_cfg.impedance_mode == "variable":
            damping_ratio = torch.as_tensor(controller_cfg.motion_damping_ratio_task, device=actions.device)
            actions[:, action_offset + command_offset : action_offset + command_offset + 6] = (
                damping_ratio / action_cfg.damping_ratio_scale
            )

        action_offset += term_dim


def _run_environments(
    task_name,
    device,
    num_envs,
    num_steps=20,
    multi_agent=False,
    create_stage_in_memory=False,
    disable_clone_in_fabric=False,
    physics_preset_name: str | None = None,
):
    """Run all environments and check environments return valid signals.

    Args:
        task_name: Name of the environment.
        device: Device to use (e.g., 'cuda').
        num_envs: Number of environments.
        num_steps: Number of simulation steps.
        multi_agent: Whether the environment is multi-agent.
        create_stage_in_memory: Whether to create stage in memory.
        disable_clone_in_fabric: Whether to disable fabric cloning.
        physics_preset_name: Name of the physics preset to apply (e.g., 'newton_mjwarp').
            If None, uses the environment's default physics.
    """

    # skip test if stage in memory is not supported
    if create_stage_in_memory and get_isaac_sim_version().major < 5:
        pytest.skip("Stage in memory is not supported in this version of Isaac Sim")

    # skip these environments as they cannot be run with 32 environments within reasonable VRAM
    if num_envs == 32 and task_name in [
        "IsaacContrib-Stack-Cube-Franka-IK-Rel-Blueprint",
        "IsaacContrib-Stack-Cube-Instance-Randomize-Franka-IK-Rel",
        "IsaacContrib-Stack-Cube-Instance-Randomize-Franka",
        "IsaacContrib-PickPlace-G1-InspireFTP-Abs",
    ]:
        return

    # skip these environments as they cannot be run with 32 environments within reasonable VRAM
    if "Visuomotor" in task_name and num_envs == 32:
        return

    print(f""">>> Running test for environment: {task_name}""")
    _check_random_actions(
        task_name,
        device,
        num_envs,
        num_steps=num_steps,
        multi_agent=multi_agent,
        create_stage_in_memory=create_stage_in_memory,
        disable_clone_in_fabric=disable_clone_in_fabric,
        physics_preset_name=physics_preset_name,
    )
    print(f""">>> Closing environment: {task_name}""")
    print("-" * 80)


def _check_random_actions(
    task_name: str,
    device: str,
    num_envs: int,
    num_steps: int = 20,
    multi_agent: bool = False,
    create_stage_in_memory: bool = False,
    disable_clone_in_fabric: bool = False,
    physics_preset_name: str | None = None,
):
    """Run random actions and check environments return valid signals.

    Args:
        task_name: Name of the environment.
        device: Device to use (e.g., 'cuda').
        num_envs: Number of environments.
        num_steps: Number of simulation steps.
        multi_agent: Whether the environment is multi-agent.
        create_stage_in_memory: Whether to create stage in memory.
        disable_clone_in_fabric: Whether to disable fabric cloning.
        physics_preset_name: Name of the physics preset to apply (e.g., 'newton_mjwarp').
            If None, uses the environment's default physics.
    """
    # create a new context stage, if stage in memory is not enabled
    if not create_stage_in_memory:
        sim_utils.create_new_stage()

    # reset the rtx sensors setting to False
    get_settings_manager().set_bool("/isaaclab/render/rtx_sensors", False)
    env = None
    try:
        # Parse the requested physics preset before resolving the config. ``parse_env_cfg`` resolves every preset to
        # its default, so applying a physics override afterwards cannot replace the already-resolved configuration.
        if physics_preset_name is not None:
            env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
            env_cfg = resolve_presets(env_cfg, selected=(physics_preset_name,))
            env_cfg.sim.device = device
            if num_envs is not None:
                env_cfg.scene.num_envs = num_envs
        else:
            env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
        reset_event = getattr(env_cfg.events, "reset_strategies", None)
        if reset_event is not None and "state_table_size" in reset_event.params:
            reset_event.params["state_table_size"] = min(32, reset_event.params["state_table_size"])
        # set config args
        env_cfg.sim.create_stage_in_memory = create_stage_in_memory
        if disable_clone_in_fabric:
            env_cfg.scene.clone_in_fabric = False

        # filter based off multi agents mode and create env
        if multi_agent:
            if not hasattr(env_cfg, "possible_agents"):
                print(f"[INFO]: Skipping {task_name} as it is not a multi-agent task")
                return
        else:
            if hasattr(env_cfg, "possible_agents"):
                print(f"[INFO]: Skipping {task_name} as it is a multi-agent task")
                return

        # TODO: Selecting the MJWarp preset routes through the Newton backend, which does not yet
        # support multi-asset spawning; some combinations fail config validation here with a
        # ValueError. Consider filtering invalid combinations in setup_environment() rather than
        # forgiving them at runtime. See PR #5097 commit fb2c74a3862 for a workaround that caught
        # the error and called pytest.skip().
        env = gym.make(task_name, cfg=env_cfg)

        # disable control on stop
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore

        # reset environment
        obs, _ = env.reset()

        # check signal
        assert _check_valid_tensor(obs)

        _fire_all_interval_events_once(env)

        # simulate environment for num_steps
        with torch.inference_mode():
            for _ in range(num_steps):
                # sample actions according to the defined space
                if multi_agent:
                    actions = {
                        agent: sample_space(
                            env.unwrapped.action_spaces[agent], device=env.unwrapped.device, batch_size=num_envs
                        )
                        for agent in env.unwrapped.possible_agents
                    }
                else:
                    actions = sample_space(
                        env.unwrapped.single_action_space, device=env.unwrapped.device, batch_size=num_envs
                    )
                    _configure_osc_smoke_actions(env, actions)
                # apply actions
                transition = env.step(actions)
                # check signals
                for data in transition[:-1]:  # exclude info
                    if multi_agent:
                        for agent, agent_data in data.items():
                            assert _check_valid_tensor(agent_data), f"Invalid data ('{agent}'): {agent_data}"
                    else:
                        assert _check_valid_tensor(data), f"Invalid data: {data}"

    finally:
        # Always ensure cleanup happens, regardless of success or failure
        if env is not None:
            env.close()

        # Clear the simulation context singleton (also closes the USD context stage)
        SimulationContext.clear_instance()


def _check_valid_tensor(data: torch.Tensor | dict) -> bool:
    """Checks if given data does not have corrupted values.

    Args:
        data: Data buffer.

    Returns:
        True if the data is valid.
    """
    if isinstance(data, torch.Tensor):
        return not torch.any(torch.isnan(data))
    elif isinstance(data, (tuple, list)):
        return all(_check_valid_tensor(value) for value in data)
    elif isinstance(data, dict):
        return all(_check_valid_tensor(value) for value in data.values())
    else:
        raise ValueError(f"Input data of invalid type: {type(data)}.")
