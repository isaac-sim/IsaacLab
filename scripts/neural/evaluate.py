# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate policies that were trained on the neural robot dynamics model of a task, in the physics it learned from.

For each physics preset in ``--trained_on``, by default the preset of the model and the preset it learned from, the
script loads the policy of the latest RSL-RL run that was trained on that physics. It runs the policy under the play
settings of the task, on the physics the model learned from and on the robot the model stands for, which is the task
without the events that the recipe names as unseen by the model. Over the first episode of every environment it
reports the reward, the length, how the episodes end, and the command metrics that the task logs. Every policy
starts from the same seed. A policy that learned on a good model scores like the one that learned on the solver.

Settings of the task can be set on the command line as ``env.<path>=value``, and another physics to run on as
``physics=<preset>``, for example the preset of the model itself.

Run from the repository root:

    uv run --extra newton-neural python scripts/neural/evaluate.py --task Isaac-Reach-Franka
"""

# Disable Warp backward pass before kernel imports (matches Isaac Lab CLI behavior).
import warp as wp

wp.config.enable_backward = False

import argparse
import re
from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import torch
import yaml
from rsl_rl.runners import OnPolicyRunner
from torch import Tensor

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import to_dict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, resolve_task_config, setup_preset_cli

RECIPE_ENTRY_POINT = "nerd_recipe_cfg_entry_point"
AGENT_ENTRY_POINT = "rsl_rl_cfg_entry_point"
PHYSICS_SELECTORS = ("physics=", "env.sim.physics=")  # Overrides that name the physics preset of a task


class SettingsLoader(yaml.SafeLoader):
    """Reads the settings that a run saved, without building the Python objects among them."""


SettingsLoader.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: None)


def physics_classes(physics: dict) -> tuple[str | None, str | None]:
    """Return the classes of the physics manager and of its solver, as a saved configuration names them."""
    return physics.get("class_type"), (physics.get("solver_cfg") or {}).get("class_type")


def trained_run(folder: Path, physics: tuple[str | None, str | None]) -> Path | None:
    """Return the latest run in ``folder`` that saved a policy and whose task settings name ``physics``, or ``None``.

    Runs on different physics can share an experiment folder, so the settings that every run saves decide.
    """
    for run in sorted(folder.iterdir(), reverse=True):
        if not (run / "params" / "env.yaml").is_file() or not any(run.glob("model_*.pt")):
            continue
        with open(run / "params" / "env.yaml") as file:
            settings = yaml.load(file, Loader=SettingsLoader)
        if physics_classes(settings["sim"]["physics"] or {}) == physics:
            return run
    return None


@torch.inference_mode()
def score(
    env: RslRlVecEnvWrapper, policy: Callable[..., Tensor]
) -> tuple[int, float, float, dict[str, float], dict[str, float]]:
    """Run ``policy`` for the length of one episode and average over the first episode of every environment.

    An environment that ends early starts another episode. Counting that one too would count early endings more
    often than they happen, so every environment counts once.

    Returns:
        The number of episodes, their mean reward and mean length [s], the fraction of them that each termination
        term ended, and the mean of every command metric that the task logs at the end of an episode, keyed
        ``<command>/<metric>``. The task logs a metric as its mean over all environments that reset in a step, so
        its mean here is that of the steps on which first episodes end, weighted by their number. Without an episode
        that ended, the means are ``nan``.
    """
    task = env.unwrapped
    obs, _ = env.reset()
    reward, seconds = (torch.zeros(task.num_envs, device=task.device) for _ in range(2))
    running = torch.ones(task.num_envs, dtype=torch.bool, device=task.device)  # Environments in their first episode
    endings = dict.fromkeys(task.termination_manager.active_terms, 0)
    metrics: dict[str, float] = {}
    for _ in range(task.max_episode_length):
        obs, step_reward, dones, extras = env.step(policy(obs))
        if hasattr(policy, "reset"):
            policy.reset(dones)  # Clears the memory of a recurrent policy in the environments that ended
        reward += step_reward * running
        seconds += task.step_dt * running
        ended = running & dones.bool()
        running &= ~ended
        count = int(ended.sum())
        if count == 0:
            continue
        for name in endings:
            endings[name] += int((ended & task.termination_manager.get_term(name)).sum())
        for key, value in extras["log"].items():
            if key.startswith("Metrics/"):
                name = key.removeprefix("Metrics/")
                metrics[name] = metrics.get(name, 0.0) + float(value) * count
        if not running.any():
            break
    episode_count = int((~running).sum())
    divisor = episode_count or float("nan")
    return (
        episode_count,
        reward[~running].sum().item() / divisor,
        seconds[~running].sum().item() / divisor,
        {name: count / divisor for name, count in endings.items()},
        {name: total / divisor for name, total in metrics.items()},
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="Task whose policies are evaluated.")
    parser.add_argument("--num_envs", type=int, default=1024, help="Environments that run in parallel.")
    parser.add_argument(
        "--trained_on",
        nargs="+",
        default=None,
        help="Physics presets the policies were trained with. Overrides go before this option.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed of the environment.")
    add_launcher_args(parser)
    args, overrides = setup_preset_cli(parser, argv)

    if RECIPE_ENTRY_POINT not in gym.spec(args.task).kwargs:
        parser.error(
            f"the task {args.task} registers no recipe for a neural robot dynamics model. Add an "
            f"isaaclab_neural.NeRDRecipeCfg to its gym registration as {RECIPE_ENTRY_POINT}"
        )
    recipe_cfg = load_cfg_from_registry(args.task, RECIPE_ENTRY_POINT)
    presets = [recipe_cfg.physics_preset, recipe_cfg.reference_preset] if args.trained_on is None else args.trained_on
    physics = next((o.split("=", 1)[1] for o in overrides if o.startswith(PHYSICS_SELECTORS)), None)
    if physics is None:
        physics = recipe_cfg.reference_preset
        overrides = [f"physics={physics}", *overrides]
    try:
        env_cfg, _ = resolve_task_config(args.task, AGENT_ENTRY_POINT, play_mode=True, overrides=overrides)
        trained = {
            preset: resolve_task_config(args.task, AGENT_ENTRY_POINT, overrides=[f"physics={preset}"])
            for preset in presets
        }
    except ValueError as error:
        parser.error(str(error))
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    # The policies run the robot that the model stands for, without the randomization it does not see.
    events = env_cfg.events
    terms = {} if events is None else events if isinstance(events, dict) else vars(events)
    unknown = [name for name in recipe_cfg.unseen_events if name not in terms]
    if unknown:
        parser.error(
            f"the recipe of {args.task} names {', '.join(unknown)} in unseen_events, and the task has no event of "
            "that name"
        )
    for name in recipe_cfg.unseen_events:
        terms[name] = None

    # The policies are looked up before the simulation starts. The agent settings of a preset give the folder of its
    # runs and the clipping of their actions, and the task settings that a run saved tell its physics.
    policies = {}
    for preset, (trained_cfg, agent_cfg) in trained.items():
        folder = Path("logs/rsl_rl") / agent_cfg.experiment_name
        run = trained_run(folder, physics_classes(to_dict(trained_cfg.sim)["physics"])) if folder.is_dir() else None
        if run is None:
            print(f"No policy trained with physics={preset} yet, {folder} holds no run on it that saved a policy.")
            continue
        # The helper takes the name of the run as a pattern.
        run_pattern = re.escape(run.name) + "$"
        policies[preset] = agent_cfg, get_checkpoint_path(str(folder.resolve()), run_pattern, r"model_.*\.pt")
    if not policies:
        parser.error(f"no policy of {args.task} to evaluate. Train one with isaaclab train and physics=<preset>")

    with launch_simulation(env_cfg, args):
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
        try:
            device = env.unwrapped.device
            for preset, (agent_cfg, checkpoint) in policies.items():
                env.clip_actions = agent_cfg.clip_actions
                runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=device)
                runner.load(checkpoint, map_location=device)
                env.unwrapped.seed(args.seed)  # Every policy meets the same random draws
                episode_count, reward, seconds, endings, metrics = score(env, runner.get_inference_policy(device))
                print(f"Policy trained with physics={preset}, {checkpoint}")
                if episode_count == 0:
                    print(f"  On physics={physics} no episode ended within the length of one episode.")
                    continue
                print(
                    f"  On physics={physics} an episode collects a reward of {reward:.3f} and lasts {seconds:.2f} s, "
                    f"on average over the first episode of {episode_count} environments."
                )
                print("  Episodes end by " + ", ".join(f"{name} {share:.1%}" for name, share in endings.items()) + ".")
                for name, value in metrics.items():
                    print(f"  {name} is {value:.4f} on average over the episodes.")
        finally:
            env.close()


if __name__ == "__main__":
    main()
