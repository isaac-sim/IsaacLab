# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environments that run an MDP2 task on an interactive scene, in place of the managers of manager-based ones."""

from __future__ import annotations

import copy
import functools
import itertools
import math
from collections.abc import Callable
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics.newton_backend import capture_graph

from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils import use_stage
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.warp import ProxyArray

from isaaclab_experimental import mdp2
from isaaclab_experimental.mdp2 import rng
from isaaclab_experimental.mdp2.capture import torch_in_warp_capture
from isaaclab_experimental.mdp2.plan import Call

from .mdp2_env_cfg import MDP2EnvCfg, MDP2RLEnvCfg, ResetStep, Stage, Step
from .mdp2_scene_data import InteractiveSceneData
from .mdp2_verify import verify_capture


class MDP2Env:
    """An interactive scene driven by an MDP2 task, like a :class:`~isaaclab.envs.ManagerBasedEnv` without managers.

    A control step applies the actions, steps the physics ``decimation`` times, evaluates the MDP, resets the
    worlds whose episodes ended, and observes them, in the order of the config's ``step`` and ``reset_step``, with
    their :class:`~.mdp2_env_cfg.Stage` s. Resets are masks over the worlds, so every step after the first replays as
    CUDA graphs, physics included: one graph for a fully capturable step, and otherwise one per run of capturable
    work, with the work that cannot be captured run between them. Newton only.
    """

    diagnostics: ClassVar[bool] = False  # Whether evaluate writes the reward and termination of every term
    final_observations: bool = False  # Whether evaluate writes the observations of the worlds that ended

    def __init__(self, cfg: MDP2EnvCfg) -> None:
        _check_step(cfg)
        if cfg.verify_capture:
            # Two throwaway runs of the task, before this one exists: there is one simulation at a time.
            verify_capture(type(self), cfg, cfg.verify_capture)
        self.cfg = cfg
        # The MDP's random streams start from the seed that was set, also when -1 asks for a random one.
        self._seed = 0 if cfg.seed is None else configure_seed(cfg.seed)
        if SimulationContext.instance() is not None:
            raise RuntimeError("A simulation context already exists. Close the environment that created it first.")

        self.sim = SimulationContext(cfg.sim)
        try:
            self._init_sim()
        except Exception:
            self.sim.clear_instance()
            raise
        self._is_closed = False

    def _init_sim(self) -> None:
        """Build the scene and the MDP; on failure, the caller clears the simulation context."""
        cfg, sim = self.cfg, self.sim

        # Terms read sensor data in kernels, so sensors refresh every step rather than when their data is read. A
        # shallow copy keeps the entities that were added to the config as attributes, and the caller's config.
        scene_cfg = copy.copy(cfg.scene)
        scene_cfg.lazy_sensor_update = False
        with use_stage(sim.stage):
            self.scene = InteractiveScene(scene_cfg)
        sim.register_interactive_scene(self.scene)
        with use_stage(sim.stage):
            sim.reset()
        self.scene.update(dt=self.physics_dt)

        sim.backend_manager.set_decimation(cfg.decimation)
        # Newton-native actuators step all substeps in one call.
        self._substeps = cfg.decimation if sim.backend_manager.handles_decimation() else 1
        # Newton actuators that it cannot capture, such as networks loaded from Torch checkpoints, run on the host:
        # the physics, which steps them, and the scene reset, which resets them, then run between the graphs.
        actuators = sim.backend_manager.backend.actuators
        self._physics_capturable = actuators is None or actuators.is_all_graphable

        self.scene_data = InteractiveSceneData(self.scene, sim, self.step_dt)
        schema = self.scene_data.schema
        builder = mdp2.MDPBuilder(schema)
        if isinstance(cfg.task, mdp2.TaskCfg):
            builder.add_task(cfg.task)
        else:
            # The first tasks get one more world when the worlds do not split evenly.
            for index, task in enumerate(cfg.task):
                task_builder = mdp2.MDPBuilder(schema)
                task_builder.add_task(task)
                num_worlds = schema.num_worlds // len(cfg.task) + (index < schema.num_worlds % len(cfg.task))
                builder.replicate(task_builder, num_worlds if cfg.task_worlds is None else cfg.task_worlds[index])
        self.mdp = builder.finalize(
            device=self.device,
            backend=cfg.backend,
            diagnostics=self.diagnostics,
            final_observations=self.final_observations,
        )
        self.state = self.mdp.state(self.scene_data, seed=self._seed)
        self._all_envs = ProxyArray(wp.ones(self.num_envs, dtype=wp.bool, device=self.device))
        mdp2.startup(self.mdp, self.state)
        if self.mdp.startup_writes:
            # Startup events write scene fields in place, which the scene commits to the solver.
            self.scene_data.commit(self._all_envs.warp, self.mdp.startup_writes)
        self.transition = self.mdp.transition()

        self.actions = ProxyArray(wp.zeros((self.num_envs, self.mdp.action_dim), dtype=wp.float32, device=self.device))
        # The worlds whose scene writes a step commits: those reset, and those where an interval event wrote.
        self._interval_worlds = self.state.arrays.get("interval_worlds")
        if self._interval_worlds is not None:
            self._commit_mask = ProxyArray(wp.zeros(self.num_envs, dtype=wp.bool, device=self.device))
        self.common_step_counter = 0
        self.extras: dict[str, Any] = {}

        self._work = self._step_work()  # The work of a step, in order, each with whether a graph can capture it
        self._capturable = cfg.capture and wp.get_device(self.device).is_cuda
        self._runs: list[Callable[[], None]] | None = None  # The captured step: graph replays and the work between them
        self.num_graphs = 0  # CUDA graphs that one captured step replays
        self._pool = torch.cuda.MemPool() if self._capturable else None  # Torch memory that the graphs use
        self._stepped = False

    @property
    def num_envs(self) -> int:
        return self.scene.num_envs

    @property
    def device(self) -> str:
        return self.sim.device

    @property
    def physics_dt(self) -> float:
        return self.cfg.sim.dt

    @property
    def step_dt(self) -> float:
        return self.cfg.sim.dt * self.cfg.decimation

    @property
    def is_captured(self) -> bool:
        """Whether steps replay as CUDA graphs."""
        return self._runs is not None

    @property
    def observations(self) -> VecEnvObs:
        """The observations of every group, as Torch views of buffers that each step overwrites."""
        return {group: values.torch for group, values in self.transition.observations.items()}

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[VecEnvObs, dict]:
        """Reset every world and return their observations. A seed also reseeds the MDP's random streams."""
        if seed is not None:
            seed = configure_seed(seed)
            # In place: captured graphs hold the buffer's address.
            self.state.rng.torch.copy_(rng.streams(seed, self.mdp.streams, self.num_envs, self.device))
        for _, work in self._reset_work(self._all_envs):
            work()
        return self.observations, self.extras

    def step(self, action: torch.Tensor) -> tuple[VecEnvObs, dict]:
        """Apply the actions, step the physics, reset the worlds whose episodes ended, and observe."""
        self._step_or_replay(action)
        return self.observations, self.extras

    def close(self) -> None:
        if not self._is_closed:
            # The captured steps reference scene buffers and solver hooks.
            self._runs = None
            self.sim.stop()
            self.scene.close()
            del self.state, self.transition, self.scene_data, self.scene
            self.sim.clear_instance()
            self._is_closed = True

    def _step_or_replay(self, action: torch.Tensor) -> None:
        """Run a control step: eagerly the first time, then as CUDA graphs. Render after it if rendering is on."""
        if action.shape != self.actions.torch.shape:
            raise ValueError(f"Expected actions of shape {tuple(self.actions.torch.shape)}, got {tuple(action.shape)}.")
        self.actions.torch.copy_(action)

        if self._runs is None and self._capturable and self._stepped:
            # The first step loaded every kernel module, so the capture records launches only.
            self._runs = self._capture_step()
        if self._runs is not None:
            for run in self._runs:
                run()
        else:
            for _, work in self._work:
                work()

        self._stepped = True
        self.common_step_counter += 1

        # Rendering runs on the host, outside the graphs, so at most once per control step, after it: when one of
        # its physics steps reaches the render interval, as in MDP1.
        interval, decimation = self.cfg.sim.render_interval, self.cfg.decimation
        physics_steps = self.common_step_counter * decimation
        if physics_steps // interval > (physics_steps - decimation) // interval and self.sim.is_rendering:
            self.sim.render()

    def _step_work(self) -> list[tuple[bool, Callable[[], None]]]:
        """Return the work of a control step, in the order of the config's ``step``, each with whether a CUDA graph
        can capture it."""
        transition = self.transition
        evaluate = Call(self.state, transition=transition, observations=transition.observations)
        work = []
        for item in self.cfg.step:
            if item is Step.PHYSICS:
                work.append((self._physics_capturable, self._simulate))
            elif item is Step.EVALUATE:
                work += [
                    (launch.capturable, functools.partial(launch, evaluate)) for launch in self.mdp.launches.evaluate
                ]
                work.append((True, self._after_evaluate))
            elif item is Step.RESET:
                work += self._reset_work(transition.done)
            else:
                work.append(self._stage_work(item))
        return work

    def _reset_work(self, mask: ProxyArray) -> list[tuple[bool, Callable[[], None]]]:
        """Return the work of resetting the masked worlds, in the order of the config's ``reset_step``."""
        launches = self.mdp.launches
        commit = functools.partial(self.scene_data.commit, mask.warp, self.mdp.reset_writes)
        reset = Call(self.state, mask=mask, commit=commit)
        observe = Call(self.state, mask=mask, observations=self.transition.observations)
        work = []
        for item in self.cfg.reset_step:
            if item is ResetStep.EVENTS:
                work.append((self._physics_capturable, functools.partial(self._reset_scene, mask)))
                work += [(launch.capturable, functools.partial(launch, reset)) for launch in launches.reset]
            elif item is ResetStep.COMMIT:
                work.append((True, functools.partial(self._commit, mask)))
            elif item is ResetStep.OBSERVE:
                work += [(launch.capturable, functools.partial(launch, observe)) for launch in launches.observe]
            else:
                work.append(self._stage_work(item))
        return work

    def _stage_work(self, stage: Stage) -> tuple[bool, Callable[[], None]]:
        """Return a stage bound to the environment, followed by the simulation's forward, which passes model and state
        changes that it made through the assets to the solver. Inside a graph, forward records nothing when there are
        none."""
        return stage.capturable, functools.partial(self._run_stage, stage.fn)

    def _run_stage(self, fn: Callable[[MDP2Env], None]) -> None:
        fn(self)
        self.sim.forward()

    def _capture_step(self) -> list[Callable[[], None]]:
        """Capture each run of capturable work of a step as one graph, and bind the other work between them."""
        runs = []
        for capturable, group in itertools.groupby(self._work, lambda item: item[0]):
            steps = [step for _, step in group]
            if not capturable:
                # Eager work may change the model or the state through the assets, such as new masses, which the
                # simulation applies when it next steps on the host; replayed graphs do not, so apply them here.
                runs += [*steps, self.sim.forward]
                continue

            def run_all(steps: list[Callable[[], None]] = steps) -> None:
                for step in steps:
                    step()

            # Torch launches, of either backend, join the Warp capture.
            with torch_in_warp_capture(self.device, self._pool):
                graph = capture_graph(self.device, run_all)
            runs.append(functools.partial(wp.capture_launch, graph))
            self.num_graphs += 1

        return runs

    def _after_evaluate(self) -> None:
        """Read the step's results before the worlds that ended are reset. Part of the captured step."""

    def _simulate(self) -> None:
        """Apply the actions and step the physics ``decimation`` times."""
        # Actions apply before every physics step, as in MDP1, so targets relative to the joint state follow it.
        for _ in range(self.cfg.decimation // self._substeps):
            mdp2.apply_actions(self.mdp, self.state, self.actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt * self._substeps)

    def _commit(self, mask: ProxyArray) -> None:
        """Commit the writes of the reset events in the masked worlds, and of the interval events of the last step in
        the worlds where they ran. Part of the captured step."""
        if self._interval_worlds is None:
            self.scene_data.commit(mask.warp, self.mdp.reset_writes)
            return
        torch.logical_or(mask.torch, self._interval_worlds.torch, out=self._commit_mask.torch)
        self.scene_data.commit(self._commit_mask.warp, self.mdp.reset_writes | self.mdp.interval_writes)

    def _reset_scene(self, mask: ProxyArray) -> None:
        """Reset the assets and sensors of the masked worlds, before the MDP's reset events."""
        assets = [
            *self.scene.articulations.values(),
            *self.scene.rigid_objects.values(),
            *self.scene.surface_grippers.values(),
        ]
        for asset in assets:
            asset.reset(env_mask=mask.warp)
        # Clear sensor histories, so that a new episode does not see the contacts that ended the last one.
        for sensor in self.scene.sensors.values():
            sensor.reset(env_mask=mask.warp)


def _check_step(cfg: MDP2EnvCfg) -> None:
    """Raise if the config's step or reset step leaves out, repeats, or reorders its built-in work."""
    orders = (
        (
            "step",
            cfg.step,
            list(Step),
            "evaluation reads the state that the physics reached, and the reset restarts the worlds that it ended",
        ),
        (
            "reset_step",
            cfg.reset_step,
            list(ResetStep),
            "the commit passes the writes of the reset events to the"
            " simulation, and the observations of the reset worlds read them",
        ),
    )
    for name, items, order, reason in orders:
        builtin = [item for item in items if not isinstance(item, Stage)]
        if builtin != order:
            raise ValueError(
                f"MDP2EnvCfg.{name} must hold {', '.join(map(str, order))} once each, in this order, with stages"
                f" anywhere between them, got {[str(item) for item in builtin]}: {reason}."
            )


class MDP2RLEnv(MDP2Env, gym.Env):
    """A Gym environment driven by an MDP2 task, like a :class:`~isaaclab.envs.ManagerBasedRLEnv` without managers.

    :meth:`step` returns copies of the observations after resets, and the rewards and the terminated and truncated
    flags of the step as Torch views of buffers that the next step overwrites. A world is truncated only if it did not
    terminate. ``extras["log"]`` holds, per term, the average over worlds of their last finished episode: the reward
    sum per second, whether the term ended it, and the value that each logged quantity had at its end. With
    :attr:`~MDP2RLEnvCfg.compute_final_obs`, ``extras["final_obs"]`` holds the observations that the worlds that
    ended had before their reset, also as views; only the rows of the worlds that ended are of this step.
    """

    is_vector_env: ClassVar[bool] = True
    diagnostics: ClassVar[bool] = True
    metadata: ClassVar[dict[str, Any]] = {"render_modes": [None]}

    cfg: MDP2RLEnvCfg

    def __init__(self, cfg: MDP2RLEnvCfg, render_mode: str | None = None, **kwargs: Any) -> None:
        self.final_observations = cfg.compute_final_obs
        super().__init__(cfg)
        if cfg.compute_final_obs:
            self.extras["final_obs"] = {
                group: values.torch for group, values in self.transition.final_observations.items()
            }
        self.render_mode = render_mode
        self.metadata = {**self.metadata, "render_fps": 1 / self.step_dt}

        self.single_observation_space = gym.spaces.Dict(
            {
                group: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,))
                for group, dim in self.mdp.observation_dims.items()
            }
        )
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.mdp.action_dim,))
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        # Per-term episode statistics, kept on the device so that logging needs no host sync.
        self._reward_names = [op.name.split(".", 1)[1] for op in self.mdp.ops[mdp2.Phase.REWARDS]]
        self._termination_names = [op.name.split(".", 1)[1] for op in self.mdp.ops[mdp2.Phase.TERMINATIONS]]
        # Terms are averaged over the worlds where they are active.
        self._active_worlds = {
            phase: torch.tensor(
                [self.num_envs if op.worlds is None else len(op.worlds) for op in self.mdp.ops[phase]],
                device=self.device,
            )
            for phase in (mdp2.Phase.REWARDS, mdp2.Phase.TERMINATIONS)
        }
        self._episode_rewards = torch.zeros(self.num_envs, len(self._reward_names), device=self.device)
        self._last_episode_rewards = torch.zeros_like(self._episode_rewards)
        self._last_terminations = torch.zeros(self.num_envs, len(self._termination_names), device=self.device)
        self._metrics = {
            op.name.split(".", 1)[1]: (
                self.state.arrays[op.writes[0]].torch,
                torch.zeros(self.num_envs, device=self.device),
                slice(None) if op.worlds is None else list(op.worlds),
            )
            for op in self.mdp.ops[mdp2.Phase.QUANTITIES]
            if op.log
        }
        # The time limit is not an op: whether it ended the last episode of each world is kept apart, under its name.
        tasks = [cfg.task] if isinstance(cfg.task, mdp2.TaskCfg) else cfg.task
        time_limits = [
            name for task in tasks for name, term in task.terminations.items() if isinstance(term, mdp2.TimeLimit)
        ]
        self._time_limit = time_limits[0] if time_limits else None
        self._last_time_outs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.obs_buf: VecEnvObs = {}

    @property
    def max_episode_length(self) -> int | float:
        """Control steps until a time-out, from the task's time limit."""
        return self.mdp.max_episode_length or math.inf

    @property
    def max_episode_length_s(self) -> float:
        return self.max_episode_length * self.step_dt

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Control steps since each world's last reset."""
        return self.state.episode_length.torch

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        # Copy in place: captured graphs hold the buffer's address.
        self.state.episode_length.torch.copy_(value)

    @staticmethod
    def seed(seed: int = -1) -> int:
        return configure_seed(seed)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[VecEnvObs, dict]:
        """Reset every world and return copies of their observations."""
        super().reset(seed, options)
        self._episode_rewards.zero_()

        # Learners keep the observations across the next step, which overwrites the buffers.
        self.obs_buf = {group: values.clone() for group, values in self.observations.items()}
        return self.obs_buf, self.extras

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Apply the actions, step the physics, evaluate the MDP, reset the worlds that ended, and observe."""
        self._step_or_replay(action)
        self._log_episodes()
        self.obs_buf = {group: values.clone() for group, values in self.observations.items()}

        transition = self.transition
        return (
            self.obs_buf,
            transition.rewards.torch,
            transition.terminated.torch,
            transition.truncated.torch,
            self.extras,
        )

    def _after_evaluate(self) -> None:
        """Keep the logged quantities of the worlds that ended, and whether they timed out, before their reset."""
        done = self.transition.done.torch
        for value, last, _ in self._metrics.values():
            torch.where(done, value, last, out=last)
        if self._time_limit is not None:
            timed_out = self.state.episode_length.torch >= self.max_episode_length
            torch.where(done, timed_out, self._last_time_outs, out=self._last_time_outs)

    def _log_episodes(self) -> None:
        """Add the step's term rewards to the episode sums, and record the episodes that ended in ``extras["log"]``."""
        transition = self.transition
        ended = transition.done.torch[:, None]
        self._episode_rewards += transition.reward_terms.torch

        torch.where(ended, self._episode_rewards, self._last_episode_rewards, out=self._last_episode_rewards)
        terminations = transition.termination_terms.torch.float()
        torch.where(ended, terminations, self._last_terminations, out=self._last_terminations)
        self._episode_rewards.masked_fill_(ended, 0.0)

        rewards = self._last_episode_rewards.sum(dim=0) / self._active_worlds[mdp2.Phase.REWARDS]
        rewards /= self.max_episode_length_s
        terminations = self._last_terminations.sum(dim=0) / self._active_worlds[mdp2.Phase.TERMINATIONS]
        log = {f"Episode_Reward/{name}": rewards[k] for k, name in enumerate(self._reward_names)}
        log |= {f"Episode_Termination/{name}": terminations[k] for k, name in enumerate(self._termination_names)}
        if self._time_limit is not None:
            log[f"Episode_Termination/{self._time_limit}"] = self._last_time_outs.float().mean()
        # A quantity that only some worlds compute is averaged over them.
        for name, (_, last, worlds) in self._metrics.items():
            log[f"Metrics/{name}"] = last[worlds].mean()
        self.extras["log"] = log
