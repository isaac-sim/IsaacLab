# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a DisplayPort cable-insertion task-space policy for Isaac ROS Deploy.

The generic exporter (``export.py``) publishes the actor observation as a single opaque
vector and the raw ``[-1, 1]`` policy output as the action. The Isaac ROS Deploy bridge
instead expects the task-space contract: four named pose tensors in, and the clipped,
scaled Cartesian pose delta out.

Rather than adding task-specific branches to the generic entry point, this script reuses
it wholesale and substitutes only the export routine. ``run_export_with_hydra`` resolves
``export_rsl_rl_agent`` as a module global, so rebinding that name is enough to inject
the DisplayPort contract while argument parsing, checkpoint resolution, Hydra handling,
runner construction, and graph compilation all stay shared with upstream.

Usage:

.. code-block:: bash

    ./isaaclab.sh -p scripts/reinforcement_learning/leapp/rsl_rl/export_displayport_insertion.py \
        --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-ROS-Inference-v0 \
        --checkpoint logs/rsl_rl/displayport_insertion_rizon4s/<run>/model_<n>.pt \
        --task_space_contract
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from collections.abc import Mapping

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import export as _export  # noqa: E402  (generic RSL-RL exporter, reused as-is)

# Name of the exported LEAPP graph. The Deploy bridge looks the model up by this name, so
# it is deliberately independent of the (much longer) gym task id.
TASK_SPACE_EXPORT_MODEL_NAME = "DisplayPortTaskSpace"

_ROT6D_ELEMENTS = ["r00", "r01", "r02", "r10", "r11", "r12"]

# (tensor name, slice of the 18D actor observation, element names, source tag, LEAPP kind)
_TASK_SPACE_INPUT_SPEC = (
    ("eef_pos", slice(0, 3), ["x", "y", "z"], "eef_pose_pos", "state/body/position"),
    ("eef_rot_6d", slice(3, 9), _ROT6D_ELEMENTS, "eef_pose_rot6d", "state/body/rotation_6d"),
    ("socket_kp_pos", slice(9, 12), ["x", "y", "z"], "socket_kp_pose_pos", "state/body/position"),
    ("socket_kp_rot_6d", slice(12, 18), _ROT6D_ELEMENTS, "socket_kp_pose_rot6d", "state/body/rotation_6d"),
)

_ACTION_ELEMENT_NAMES = [
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_axis_angle_x",
    "delta_axis_angle_y",
    "delta_axis_angle_z",
]


def parse_export_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse export arguments, adding the DisplayPort task-space contract flag."""
    _leapp_scripts_dir = os.path.dirname(_THIS_DIR)
    if _leapp_scripts_dir not in sys.path:
        sys.path.insert(0, _leapp_scripts_dir)
    from export_utils import add_common_export_args, finalize_export_args

    parser = argparse.ArgumentParser(description="Export a DisplayPort cable-insertion policy with RSL-RL.")
    add_common_export_args(parser, agent_default="rsl_rl_cfg_entry_point")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder used to locate checkpoints."
    )
    parser.add_argument(
        "--task_space_contract",
        action="store_true",
        help=(
            "Export the task-space I/O contract used by Isaac ROS Deploy: split the 18D actor"
            " observation into named pose tensors (eef_pos, eef_rot_6d, socket_kp_pos,"
            " socket_kp_rot_6d) and emit the clipped, scaled Cartesian delta as 'arm_action'."
            " Requires an operational-space action term exposing position/orientation scales."
        ),
    )
    return finalize_export_args(parser, argv)


def task_space_policy_obs(obs):
    """Return the 18D actor observation tensor from an RSL-RL TensorDict."""
    if "policy" not in obs.keys():
        raise KeyError(f"Expected a 'policy' observation group, got keys: {list(obs.keys())}")
    policy_obs = obs["policy"]
    if policy_obs.shape[-1] != 18:
        raise ValueError(f"Expected 18D task-space actor observation, got shape {tuple(policy_obs.shape)}")
    return policy_obs


def split_and_annotate_task_space_obs(graph_name: str, policy_obs):
    """Expose the exact 18D trained observation as four Deploy-facing input tensors."""
    from leapp.utils.tensor_description import TensorSemantics

    parts = []
    for name, index, element_names, source, kind in _TASK_SPACE_INPUT_SPEC:
        parts.append(
            _export.annotate.input_tensors(
                graph_name,
                TensorSemantics(
                    name=name,
                    ref=policy_obs[:, index],
                    kind=kind,
                    element_names=[element_names],
                    extra={"source": source},
                ),
            )
        )
    # Re-concatenate so the policy still sees the byte-identical vector it was trained on.
    return _export.torch.cat(parts, dim=-1)


def export_task_space_action(graph_name: str, tensor, export_method: str) -> None:
    """Annotate the deploy-facing processed (clipped and scaled) task-space action."""
    from leapp.utils.tensor_description import TensorSemantics

    _export.annotate.output_tensors(
        graph_name,
        TensorSemantics(
            name="arm_action",
            ref=tensor,
            kind="target/body/pose_relative",
            element_names=[_ACTION_ELEMENT_NAMES],
            extra={
                "isaaclab_connection": "action:arm_action:pose_rel",
                "target_types": ["pose_rel"],
            },
        ),
        export_with=export_method,
    )


def task_space_action_scale(env_cfg, device, dtype):
    """Return ``[pos_scale] * 3 + [rot_scale] * 3`` from the task configuration."""
    action_cfg = env_cfg.actions.arm_action
    for attr in ("position_scale", "orientation_scale"):
        if not hasattr(action_cfg, attr):
            raise AttributeError(
                f"--task_space_contract requires an operational-space action term exposing '{attr}'; "
                f"got {type(action_cfg).__name__}."
            )
    scale_values = [float(action_cfg.position_scale)] * 3 + [float(action_cfg.orientation_scale)] * 3
    return _export.torch.tensor(scale_values, device=device, dtype=dtype).unsqueeze(0)


def export_displayport_agent(
    args_cli: argparse.Namespace,
    env_cfg,
    agent_cfg,
    simulation_app=None,
) -> bool:
    """Export a DisplayPort RSL-RL agent, optionally with the task-space I/O contract.

    Mirrors :func:`export.export_rsl_rl_agent`; the DisplayPort-specific behaviour is
    confined to the blocks guarded by ``args_cli.task_space_contract``.
    """
    _export._load_runtime_dependencies()

    task_name = args_cli.task.split(":")[-1]
    checkpoint_task_name = task_name.replace("-Play", "")

    agent_cfg = _export._update_agent_cfg_from_export_args(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1

    agent_cfg = _export.handle_deprecated_rsl_rl_cfg(agent_cfg, _export.installed_version)

    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading checkpoint search path from directory: {log_root_path}")
    if args_cli.checkpoint == "pretrained":
        backend_names = _export.get_pretrained_checkpoint_backend_names(env_cfg)
        resume_path = _export.get_published_pretrained_checkpoint("rsl_rl", checkpoint_task_name, *backend_names)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return False
    elif args_cli.checkpoint and os.path.isdir(args_cli.checkpoint):
        resume_path = _export.get_checkpoint_path(
            os.path.dirname(args_cli.checkpoint), os.path.basename(args_cli.checkpoint), agent_cfg.load_checkpoint
        )
    elif args_cli.checkpoint:
        resume_path = _export.retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = _export.get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if not resume_path:
        print(f"[INFO] No checkpoint found for task: {checkpoint_task_name} in directory: {log_root_path}")
        return False

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = None
    leapp_started = False

    try:
        env = _export.gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        policy_node_name = _export.ensure_env_spec_id(env)

        graph_name = args_cli.export_task_name if args_cli.export_task_name is not None else task_name
        if args_cli.task_space_contract and args_cli.export_task_name is None:
            graph_name = TASK_SPACE_EXPORT_MODEL_NAME
        if args_cli.task_space_contract:
            # The contract publishes inputs/outputs under ``graph_name``; the recurrent-state
            # annotations must target that same node or LEAPP cannot find it.
            policy_node_name = graph_name

        if isinstance(env.unwrapped, _export.ManagerBasedRLEnv):
            export_method = "onnx-dynamo" if args_cli.export_method is None else args_cli.export_method
            # Patch only the observation groups consumed by the actor policy.
            # This filters out the critic and teacher observation groups.
            obs_groups_cfg = getattr(agent_cfg, "obs_groups", None)
            if isinstance(obs_groups_cfg, Mapping):
                required_obs_groups = set(obs_groups_cfg.get("actor", ["policy"]))
            else:
                required_obs_groups = {"policy"}
            # The task-space contract annotates its own inputs/outputs under a dedicated graph
            # node, so the automatic patcher must stay out of the way: it would annotate the
            # same observation tensors under a node named after the task id, and LEAPP refuses
            # to mix two active tracing contexts for one tensor.
            if not args_cli.task_space_contract:
                _export.patch_env_for_export(
                    env,
                    export_method=export_method,
                    required_obs_groups=required_obs_groups,
                )
        elif args_cli.export_method is not None:
            raise ValueError(
                "--export_method is only supported for manager-based environments. For direct environments, "
                "set export_with directly in the annotate.output_tensors() call instead."
            )

        env = _export.RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = _export.OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = _export.DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)

        policy = runner.get_inference_policy(device=env.unwrapped.device)

        task_space_scale = None
        if args_cli.task_space_contract:
            task_space_scale = task_space_action_scale(
                env.unwrapped.cfg, env.unwrapped.device, next(policy.parameters()).dtype
            )
            print(f"[INFO] Exporting processed task-space action with scale: {task_space_scale.flatten().tolist()}")

        if args_cli.export_save_path is not None:
            save_path = args_cli.export_save_path
        elif args_cli.checkpoint == "pretrained":
            # Use a predictable path independent of the Nucleus mirror directory structure.
            save_path = os.path.join(".pretrained_checkpoints", "rsl_rl", checkpoint_task_name)
        else:
            save_path = log_dir
        _export.leapp.start(graph_name, save_path=save_path, max_cached_io=max(args_cli.validation_steps, 2))
        leapp_started = True
        obs = env.reset()[0]
        if simulation_app is not None:
            while not simulation_app.is_running():
                time.sleep(0.5)

        for _ in range(max(args_cli.validation_steps, 2)):
            with _export.torch.inference_mode():
                # Inputs must be annotated before ``state_tensors``: the first ``input_tensors``
                # call is what creates the graph node the recurrent state attaches to.
                if args_cli.task_space_contract:
                    policy_obs = task_space_policy_obs(obs).to(dtype=next(policy.parameters()).dtype)
                    obs_for_policy = obs.clone()
                    obs_for_policy["policy"] = split_and_annotate_task_space_obs(graph_name, policy_obs)
                else:
                    obs_for_policy = obs

                if _export.is_actor_recurrent_policy(policy):
                    actor_hidden = _export.ensure_actor_hidden_state_initialized(
                        policy,
                        batch_size=env.num_envs,
                        device=env.unwrapped.device,
                        dtype=next(policy.parameters()).dtype,
                    )
                    registered_state = _export.annotate.state_tensors(
                        policy_node_name,
                        _export.state_dict_from_actor_hidden(actor_hidden),
                    )
                    _export.set_actor_hidden_state(
                        policy, _export.actor_hidden_from_registered(registered_state, actor_hidden)
                    )

                actions = policy(obs_for_policy)

                if _export.is_actor_recurrent_policy(policy):
                    actor_hidden_after = _export.get_actor_hidden_state(policy)
                    _export.annotate.update_state(
                        policy_node_name,
                        _export.state_dict_from_actor_hidden(actor_hidden_after),
                    )

                if args_cli.task_space_contract:
                    processed_action = _export.torch.clamp(actions, -1.0, 1.0) * task_space_scale
                    export_task_space_action(graph_name, processed_action, args_cli.export_method)
                    # Refresh inputs without invoking the action manager, which would apply the
                    # raw (unscaled) action and desynchronise the traced graph.
                    obs = env.get_observations()
                else:
                    obs, _, _, _ = env.step(actions)

        _export.leapp.stop()
        leapp_started = False
        validate = args_cli.validation_steps > 0
        _export.leapp.compile_graph(visualize=not args_cli.disable_graph_visualization, validate=validate)
    finally:
        if leapp_started:
            with contextlib.suppress(Exception):
                _export.leapp.stop()
        if env is not None:
            env.close()

    return True


def main_cli(argv: list[str] | None = None) -> bool:
    """Run the DisplayPort export flow, reusing the generic Hydra/simulation wrapper."""
    args_cli, hydra_args = parse_export_args(argv)
    # ``run_export_with_hydra`` resolves ``export_rsl_rl_agent`` as a module global, so
    # rebinding it here injects the DisplayPort contract without editing the generic script.
    _export.export_rsl_rl_agent = export_displayport_agent
    return _export.run_export_with_hydra(args_cli, hydra_args)


if __name__ == "__main__":
    main_cli()
