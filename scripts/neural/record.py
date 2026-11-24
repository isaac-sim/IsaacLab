# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record the data that the neural robot dynamics model of a task is trained on.

The task names the settings of its model as ``nerd_recipe_cfg_entry_point``, an
:class:`isaaclab_neural.NeRDRecipeCfg`. The script builds the scene and simulation of the task on the physics the
model learns from, lets ``newton-neural`` restart the robots from sampled states and command joint position targets,
and steps each frame through the environment. It writes the dataset next to the file that the model's physics preset
loads, with the untrained model inside it for ``train.py``.

Settings of the recipe can be set on the command line as ``agent.<path>=value``, for example
``agent.record.episode_frames=[8,16]``, and settings of the task as ``env.<path>=value``. ``physics=<preset>``
selects another physics to learn from than the recipe names.

Run from the repository root:

    uv run --extra newton-neural python scripts/neural/record.py --task Isaac-Reach-Franka
"""

# Disable Warp backward pass before kernel imports (matches Isaac Lab CLI behavior).
import warp as wp

wp.config.enable_backward = False

import argparse
import time
from dataclasses import MISSING
from pathlib import Path

import gymnasium as gym
import newton
import torch
from isaaclab_neural import NeRDRecipeCfg, RecordingSimulation
from isaaclab_neural.physics import MJWarpNeRDSolverCfg, NeRDSolverCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from newton_neural import Frames
from newton_neural.data import TrajectoryRecorder
from newton_neural.systems import SystemArticulation

import isaaclab.envs.mdp as mdp
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass, instantiate
from isaaclab.utils.io import dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

RECIPE_ENTRY_POINT = "nerd_recipe_cfg_entry_point"
PHYSICS_SELECTORS = ("physics=", "env.sim.physics=")  # Overrides that name the physics preset of a task


@configclass
class RecordingActionsCfg:
    """The joint position targets of every joint of the robot, as they are."""

    joint_targets: mdp.JointPositionActionCfg = MISSING


@configclass
class RecordingObservationsCfg:
    """No observations. The recorder reads the state of the robot from Newton."""


@configclass
class RecordingEventsCfg:
    """No events, for a task that has none."""


@configclass
class RecordingEnvCfg(ManagerBasedEnvCfg):
    """The scene, the simulation and the startup events of a task, without its rewards, resets and commands."""

    observations: RecordingObservationsCfg = RecordingObservationsCfg()


class EnvSimulation(RecordingSimulation):
    """The robot of a recording environment, as the simulation that a ``TrajectoryRecorder`` drives.

    A frame is one environment step. Isaac Lab applies the joint position targets and steps the physics.

    Args:
        env: Environment whose action is the joint position targets of the robot.
        robot: Name of the robot in the scene.
    """

    def __init__(self, env: ManagerBasedEnv, robot: str) -> None:
        self.env, self.asset = env, env.scene[robot]
        self._backend = env.sim.backend_manager.backend
        if not self.asset.actuators.is_graphable:
            raise ValueError(
                f"Isaac Lab runs actuator models of {robot!r} in torch, which keep the joint targets from Newton's "
                "control arrays, where the recorder reads them. Run them as Newton actuators, with "
                "env_cfg.sim.use_newton_actuators = True"
            )
        self.articulation = self.asset.root_view.articulation_ids.numpy().ravel().tolist()
        system = SystemArticulation.from_model(self.model, self.articulation)
        self.binding = system.bind(self.model, self.articulation)
        self.world_count = self.binding.world_count
        self.frame_dt = env.step_dt
        self.substep_count = env.cfg.decimation * self._backend.num_substeps
        self.device = torch.device(env.device)
        # The asset takes joint values in its own joint order, the system in Newton's.
        actuated = system.spec.actuated_joint_labels
        self._joint_order = torch.tensor([actuated.index(name) for name in self.asset.joint_names], device=self.device)
        self._targets = self.binding.read_control(self.control)["joint_target_q"][:, self._joint_order].clone()

    @property
    def model(self) -> newton.Model:
        """Newton model of the environment."""
        return self._backend.model

    @property
    def state(self) -> newton.State:
        """Current Newton state. Read it again after every step, since the buffers may swap."""
        return self._backend.state_0

    @property
    def control(self) -> newton.Control:
        """Newton control with the joint targets of the last frame."""
        return self._backend.control

    @property
    def contacts(self) -> newton.Contacts | None:
        """Contacts of the current state, or ``None`` if the solver detects its own."""
        return None if self._backend.collision_pipeline is None else self._backend.contacts

    @property
    def solver(self) -> newton.solvers.SolverBase:
        """Newton solver of the environment."""
        return self._backend.solver

    @property
    def default_joint_pos(self) -> torch.Tensor:
        """Default joint positions of the robot ``[actuated_count]`` [m or rad], in the joint order of the system."""
        default = torch.empty_like(self._targets[0])
        default[self._joint_order] = self.asset.data.default_joint_pos.torch[0]
        return default

    def step(self, command: Frames | None = None) -> None:
        """Step the environment once on ``joint_target_q`` ``[world_count, actuated_count]``, or on the last targets."""
        if command is not None:
            self._targets = command["joint_target_q"][:, self._joint_order]
        self.env.step(self._targets)
        self._collide()

    def restart(self, frames: Frames, world_mask: torch.Tensor | None = None) -> None:
        """Write ``joint_q`` and ``joint_qd`` into the masked environments and reset what the scene keeps of them.

        Only the robot restarts. The other assets of the scene keep the state that the simulation left them in.
        """
        asset = self.asset
        env_mask = None if world_mask is None else wp.from_torch(world_mask, dtype=wp.bool)
        joint_q, joint_qd = frames["joint_q"], frames["joint_qd"]
        if self.binding.system.floating_base:
            # A free joint holds the pose of the root link (xyzw) and the spatial velocity of its centre of mass.
            pose = torch.cat((joint_q[:, :3], torch.nn.functional.normalize(joint_q[:, 3:7], dim=1)), dim=1)
            asset.write_root_link_pose_to_sim_mask(root_pose=pose, env_mask=env_mask)
            asset.write_root_com_velocity_to_sim_mask(root_velocity=joint_qd[:, :6], env_mask=env_mask)
            joint_q, joint_qd = joint_q[:, 7:], joint_qd[:, 6:]
        asset.write_joint_position_to_sim_mask(position=joint_q[:, self._joint_order], env_mask=env_mask)
        asset.write_joint_velocity_to_sim_mask(velocity=joint_qd[:, self._joint_order], env_mask=env_mask)
        self.env.scene.reset(env_mask=env_mask)
        # Resets the solver in the written environments and updates their body transforms.
        self.env.sim.forward()
        self._collide()

    def _collide(self) -> None:
        # The step detects contacts before it moves the robots, and the recorder needs those of the state it reads.
        backend = self._backend
        if backend.collision_pipeline is not None:
            horizon = min(backend.num_substeps, backend.cfg.collision_decimation or backend.num_substeps)
            backend.collision_pipeline.collide(backend.state_0, backend.contacts, dt=horizon * backend.solver_dt)


def recording_cfg(env_cfg: ManagerBasedRLEnvCfg, robot: str, single_solver_step: bool) -> RecordingEnvCfg:
    """Build the recording environment of a resolved task configuration.

    Args:
        env_cfg: Task configuration on the physics the model learns from.
        robot: Name of the robot in the scene.
        single_solver_step: Whether a frame is one solver step of the task instead of one environment step.
    """
    sim = env_cfg.sim
    decimation = env_cfg.decimation
    if single_solver_step:
        sim.dt, sim.physics.num_substeps = sim.dt / sim.physics.num_substeps, 1
        decimation = sim.render_interval = 1
        # MuJoCo Warp has to start every step from the Newton state, which a restart writes.
        if isinstance(sim.physics.solver_cfg, MJWarpSolverCfg):
            sim.physics.solver_cfg.update_data_interval = 1
    # Events at startup set the task up, such as its friction. Those at resets and intervals would move the robots.
    events = RecordingEventsCfg() if env_cfg.events is None else env_cfg.events
    terms = events if isinstance(events, dict) else vars(events)
    for name, term in terms.items():
        if isinstance(term, EventTermCfg) and term.mode not in ("prestartup", "startup"):
            terms[name] = None
    targets = mdp.JointPositionActionCfg(asset_name=robot, joint_names=[".*"], scale=1.0, use_default_offset=False)
    return RecordingEnvCfg(
        scene=env_cfg.scene,
        sim=sim,
        decimation=decimation,
        seed=env_cfg.seed,
        events=events,
        actions=RecordingActionsCfg(joint_targets=targets),
    )


def resolve_task(
    task: str, overrides: list[str]
) -> tuple[ManagerBasedRLEnvCfg, NeRDRecipeCfg, NeRDSolverCfg | MJWarpNeRDSolverCfg]:
    """Resolve the task on the physics its model learns from, the settings of the model, and its solver settings.

    The task comes without the events that the recipe names as unseen by the model.

    Raises:
        ValueError: If the task has no recipe or is not manager-based, if its physics is not Newton, runs a model
            itself or is not MuJoCo Warp for a model inside MuJoCo Warp, or if the recipe names a physics preset
            that runs no model or an event that the task does not have.
    """
    if RECIPE_ENTRY_POINT not in gym.spec(task).kwargs:
        raise ValueError(
            f"the task {task} registers no recipe for a neural robot dynamics model. Add an "
            f"isaaclab_neural.NeRDRecipeCfg to its gym registration as {RECIPE_ENTRY_POINT}"
        )
    # The recipe is resolved with the command line first, which can name another physics to learn from.
    _, recipe_cfg = resolve_task_config(task, RECIPE_ENTRY_POINT, overrides=overrides)
    physics = next((o.split("=", 1)[1] for o in overrides if o.startswith(PHYSICS_SELECTORS)), None)
    if physics is None:
        physics = recipe_cfg.reference_preset
        overrides = [f"physics={physics}", *overrides]
    env_cfg, recipe_cfg = resolve_task_config(task, RECIPE_ENTRY_POINT, overrides=overrides)
    recipe_cfg.reference_preset = physics  # The settings saved beside the data name the physics they came from
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(
            f"a model is recorded from the scene of a manager-based task, and {task} is configured by a "
            f"{type(env_cfg).__name__}"
        )
    source = env_cfg.sim.physics
    if not isinstance(source, NewtonCfg):
        raise ValueError(
            f"a model is recorded from Newton physics, and physics={physics} resolves to a {type(source).__name__}. "
            "Select a Newton physics preset"
        )
    if isinstance(source.solver_cfg, NeRDSolverCfg | MJWarpNeRDSolverCfg):
        raise ValueError(
            f"a model is recorded from a solver, and physics={physics} runs a model, with a "
            f"{type(source.solver_cfg).__name__}. Select the physics that the model learns from"
        )
    target, _ = resolve_task_config(task, None, overrides=[f"physics={recipe_cfg.physics_preset}"])
    solver_cfg = getattr(target.sim.physics, "solver_cfg", None)
    if not isinstance(solver_cfg, NeRDSolverCfg | MJWarpNeRDSolverCfg):
        raise ValueError(
            f"the recipe of {task} names the physics preset {recipe_cfg.physics_preset!r}, whose solver is configured "
            f"by a {type(solver_cfg).__name__}. Name a preset that runs a model, with a NeRDSolverCfg or a "
            "MJWarpNeRDSolverCfg"
        )
    if isinstance(solver_cfg, MJWarpNeRDSolverCfg) and not isinstance(source.solver_cfg, MJWarpSolverCfg):
        raise ValueError(
            f"the model of the physics preset {recipe_cfg.physics_preset!r} predicts the free acceleration inside "
            f"MuJoCo Warp and learns it from MuJoCo Warp, and physics={physics} steps with a "
            f"{type(source.solver_cfg).__name__}. Select a MuJoCo Warp preset"
        )
    # The data comes from the robot that the model stands for, without the randomization it does not see.
    if isinstance(recipe_cfg.unseen_events, str):
        raise ValueError(
            f"unseen_events of the recipe is a list of event names, and the command line gave the text "
            f"{recipe_cfg.unseen_events!r}. Write the names with quotes, as \"agent.unseen_events=['add_base_mass']\""
        )
    events = env_cfg.events
    terms = {} if events is None else events if isinstance(events, dict) else vars(events)
    unknown = [name for name in recipe_cfg.unseen_events if name not in terms]
    if unknown:
        raise ValueError(
            f"the recipe of {task} names {', '.join(unknown)} in unseen_events, and the task has no event of that name"
        )
    for name in recipe_cfg.unseen_events:
        terms[name] = None
    return env_cfg, recipe_cfg, solver_cfg


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="Task whose robot the model stands for.")
    parser.add_argument(
        "--num_envs", type=int, default=None, help="Robots recorded in parallel, the recipe's by default."
    )
    parser.add_argument("--num_frames", type=int, default=None, help="Frames per robot, the recipe's by default.")
    parser.add_argument("--dataset", type=Path, default=None, help="Dataset folder, next to the model by default.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the environment and of the recorder.")
    add_launcher_args(parser)
    args, overrides = setup_preset_cli(parser, argv)

    # What can be wrong with the request alone is reported before the simulation starts.
    try:
        env_cfg, recipe_cfg, solver_cfg = resolve_task(args.task, overrides)
    except ValueError as error:
        parser.error(str(error))
    checkpoint = Path(solver_cfg.checkpoint)
    dataset_path = checkpoint.parent / "dataset" if args.dataset is None else args.dataset
    if (
        (dataset_path / "meta.json").exists()
        or (dataset_path / ".meta.json.pending").exists()
        or any(dataset_path.glob("shard-*"))
    ):
        parser.error(f"{dataset_path} holds a dataset. Delete it, or name another folder with --dataset")
    if args.num_envs is not None:
        recipe_cfg.record.num_envs = args.num_envs
    if args.num_frames is not None:
        recipe_cfg.record.num_frames = args.num_frames
    env_cfg.scene.num_envs = recipe_cfg.record.num_envs
    env_cfg.seed = args.seed
    recipe = instantiate(recipe_cfg)
    recipe.configure(env_cfg)
    # A model inside MuJoCo Warp predicts one solver step, and a model in place of the solver one environment step.
    two_way = isinstance(solver_cfg, MJWarpNeRDSolverCfg)
    cfg = recording_cfg(env_cfg, recipe_cfg.robot, single_solver_step=two_way)

    with launch_simulation(cfg, args):
        env = ManagerBasedEnv(cfg)
        try:
            simulation = EnvSimulation(env, recipe_cfg.robot)
            nerd = recipe.model(simulation).to(simulation.device)
            if nerd.prediction.integrates == two_way:
                raise ValueError(
                    f"the recipe builds a model with a {type(nerd.prediction).__name__}, which does not fit the "
                    f"solver of the physics preset {recipe_cfg.physics_preset!r}, a {type(solver_cfg).__name__}"
                )
            recorder = TrajectoryRecorder(
                simulation,
                nerd,
                states=recipe.start_states(simulation),
                commands=recipe.joint_targets(simulation),
                episode_frames=tuple(recipe_cfg.record.episode_frames),
                seed=args.seed,
            )
            start = time.perf_counter()
            dataset = recorder.record(dataset_path, recipe_cfg.record.num_frames)
            elapsed = time.perf_counter() - start
            nerd.save(dataset.path / checkpoint.name)  # The untrained model, which train.py fits
        finally:
            env.close()
    # The settings that the data was recorded with stay next to it.
    dump_yaml(str(dataset.path / "params" / "env.yaml"), cfg)
    dump_yaml(str(dataset.path / "params" / "recipe.yaml"), recipe_cfg)
    print(
        f"Recorded {dataset.frame_count} frames of {dataset.frame_dt:.5f} s for each of {dataset.world_count} robots "
        f"in {elapsed:.1f} s. The dataset is in {dataset.path.resolve()}. Train the model with train.py next to "
        "this script."
    )


if __name__ == "__main__":
    main()
