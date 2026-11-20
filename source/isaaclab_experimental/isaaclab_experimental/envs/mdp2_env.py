# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environments that run an MDP2 task on an interactive scene, in place of the managers of manager-based ones."""

from __future__ import annotations

import copy
import dataclasses
import functools
import itertools
import math
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Literal

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics.newton_backend import capture_graph
from isaaclab_newton.renderers import NewtonOptixRendererCfg, NewtonWarpRendererCfg

from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import Camera, CameraCfg, SensorBase
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

    A control step applies the actions, steps the physics ``decimation`` times, renders the cameras, evaluates the
    MDP, resets the worlds whose episodes ended, and observes them, in the order of the config's ``step`` and
    ``reset_step``, with their :class:`~.mdp2_env_cfg.Stage` s. Resets are masks over the worlds, so every step after
    the first replays as CUDA graphs, physics and rendering included: one graph for a fully capturable step, and
    otherwise one per run of capturable work, with the work that cannot be captured run between them. Newton only.

    :attr:`observations` holds a tensor per concatenated group, and a dict of tensors, one per term with the term's
    shape and type, per group that is not concatenated.
    """

    diagnostics: ClassVar[bool] = False  # Whether evaluate writes the reward and termination of every term
    final_observations: bool = False  # Whether evaluate writes the observations of the worlds that ended
    _budgeted: bool = False  # Whether an episode budget, rather than the ended episodes, decides the worlds to reset

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
        scene_cfg.__dict__.update(_without_renderer_graphs(scene_cfg))
        with use_stage(sim.stage):
            self.scene = InteractiveScene(scene_cfg)
        sim.register_interactive_scene(self.scene)
        with use_stage(sim.stage):
            sim.reset()
        # The cameras render once per control step, the other sensors refresh with every physics step.
        scene = self.scene
        self._cameras = [sensor for sensor in scene.sensors.values() if isinstance(sensor, Camera)]
        # OptiX refits its acceleration structures on the host, which a CUDA graph cannot record: such cameras render
        # between the graphs of a step.
        self._render_capturable = not any(
            isinstance(camera.cfg.renderer_cfg, NewtonOptixRendererCfg) for camera in self._cameras
        )
        if self._cameras and Step.RENDER not in cfg.step:
            raise ValueError("The scene has cameras, so MDP2EnvCfg.step must hold Step.RENDER, or they never render.")
        sensors = [sensor for sensor in scene.sensors.values() if not isinstance(sensor, Camera)]
        # Sensors that support it refresh together, as in the scene's update.
        self._batched_sensors = [sensor for sensor in sensors if sensor.supports_batch_update]
        self._sensors = [sensor for sensor in sensors if not sensor.supports_batch_update]
        self._assets = [
            asset
            for assets in (
                scene.articulations,
                scene.rigid_objects,
                scene.rigid_object_collections,
                scene.deformable_objects,
                scene.cable_objects,
                scene.surface_grippers,
            )
            for asset in assets.values()
        ]
        self._update_scene(self.physics_dt)

        sim.backend_manager.set_decimation(cfg.decimation)
        # Newton-native actuators step all substeps in one call.
        self._substeps = cfg.decimation if sim.backend_manager.handles_decimation() else 1

        self.scene_data = InteractiveSceneData(self.scene, sim, self.step_dt)
        self._all_envs = ProxyArray(wp.ones(self.num_envs, dtype=wp.bool, device=self.device))
        self.common_step_counter = 0
        self.extras: dict[str, Any] = {}
        self._capturable = cfg.capture and wp.get_device(self.device).is_cuda
        self._pool = torch.cuda.MemPool() if self._capturable else None  # Torch memory that the graphs use
        self._build(cfg.task)

    def _build(self, task: mdp2.TaskCfg, previous: tuple[mdp2.MDP, mdp2.MDPState] | None = None) -> None:
        """Build the MDP of a task on the scene, and the step that runs it. Without a ``previous`` MDP, run its startup;
        with one, carry its state over, see :func:`~isaaclab_experimental.mdp2.migrate`."""
        builder = mdp2.MDPBuilder(self.scene_data.schema)
        builder.add_task(task)
        self.mdp = builder.finalize(
            device=self.device,
            backend=self.cfg.backend,
            diagnostics=self.diagnostics,
            final_observations=self.final_observations,
        )
        self.state = self.mdp.state(self.scene_data, seed=self._seed)
        if previous is not None:
            mdp2.migrate(*previous, self.mdp, self.state)
        else:
            mdp2.startup(self.mdp, self.state)
            if self.mdp.startup_writes:
                # Startup events write scene fields in place, which the scene commits to the solver.
                self.scene_data.commit(self._all_envs.warp, self.mdp.startup_writes)
        self.transition = self.mdp.transition()
        self._observations = self._views(self.transition.observations)
        # The worlds that a step resets: those whose episodes ended, unless an episode budget restarts only some.
        self._resets = ProxyArray(wp.zeros_like(self.transition.done.warp)) if self._budgeted else self.transition.done

        self.actions = ProxyArray(wp.zeros((self.num_envs, self.mdp.action_dim), dtype=wp.float32, device=self.device))
        # The worlds whose scene writes a step commits: those reset, and those where an interval event wrote.
        self._event_worlds = self.state.arrays.get("event_worlds")
        if self._event_worlds is not None:
            self._commit_mask = ProxyArray(wp.zeros(self.num_envs, dtype=wp.bool, device=self.device))

        self._work = self._step_work()  # The work of a step, in order, each with whether a graph can capture it
        self._runs: list[Callable[[], None]] | None = None  # The captured step: graph replays and the work between them
        self.num_graphs = 0  # CUDA graphs that one captured step replays
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
        return self._observations

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[VecEnvObs, dict]:
        """Reset every world and return their observations. A seed also reseeds the MDP's random streams."""
        if seed is not None:
            seed = configure_seed(seed)
            # In place: captured graphs hold the buffer's address.
            self.state.rng.torch.copy_(rng.streams(seed, self.mdp.streams, self.num_envs, self.device))
        for _, work in self._reset_work(self._all_envs, render=True):
            work()
        return self.observations, self.extras

    def step(self, action: torch.Tensor) -> tuple[VecEnvObs, dict]:
        """Apply the actions, step the physics, reset the worlds whose episodes ended, and observe."""
        self._step_or_replay(action)
        return self.observations, self.extras

    def update_task(self, task: mdp2.TaskCfg, truncate: bool = True) -> None:
        """Replace the task by another on the same scene, such as with a reward added or removed, and keep running.

        The state of the uses that both tasks have carries over, see :func:`~isaaclab_experimental.mdp2.migrate`: term
        state, timers, histories, commands, progress, tunable values, random streams, and reward weights. With
        ``truncate``, every world starts a new episode, so that no episode mixes two tasks; otherwise episodes go on.
        The next step captures its graphs again, which compiles the kernels of a new task once, in about a second.
        """
        self._build(task, previous=(self.mdp, self.state))
        self.cfg.task = task
        self._bind()
        if truncate:
            self.reset()

    def _bind(self) -> None:
        """Bind what an environment derives from the MDP, after it was built."""

    def tune(
        self,
        path: str,
        value: Any,
        worlds: slice | Sequence[int] | torch.Tensor | None = None,
        at: Literal["reset", "step"] = "reset",
    ) -> None:
        """Change a value of the task declared :class:`~isaaclab_experimental.mdp2.Tunable`, such as
        ``"rewards.track.weight"``, in some or all worlds, at their next reset or from the next step. Captured steps
        keep replaying. See :func:`~isaaclab_experimental.mdp2.tune`."""
        mdp2.tune(self.mdp, self.state, path, value, worlds, at)

    def close(self) -> None:
        if not self._is_closed:
            self.sim.stop()
            # The captured graphs, the MDP state, and the scene data all reference the scene's buffers.
            self._runs = None
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
                work.append((True, self._simulate))
            elif item is Step.RENDER:
                work += [(self._render_capturable, self._render)] if self._cameras else []
            elif item is Step.EVALUATE:
                work += [
                    (launch.capturable, functools.partial(launch, evaluate)) for launch in self.mdp.launches.evaluate
                ]
                work.append((True, self._after_evaluate))
            elif item is Step.RESET:
                work += self._reset_work(self._resets)
            else:
                work.append(self._stage_work(item))
        return work

    def _reset_work(self, mask: ProxyArray, render: bool = False) -> list[tuple[bool, Callable[[], None]]]:
        """Return the work of resetting the masked worlds, in the order of the config's ``reset_step``. With
        ``render``, the cameras render before the observations also without ``ResetStep.RENDER``."""
        launches = self.mdp.launches
        commit = functools.partial(self.scene_data.commit, mask.warp, self.mdp.reset_writes)
        reset = Call(self.state, mask=mask, commit=commit)
        observe = Call(self.state, mask=mask, observations=self.transition.observations)
        rendering = [(self._render_capturable, self._render)] if self._cameras else []
        render = render and ResetStep.RENDER not in self.cfg.reset_step
        work = []
        for item in self.cfg.reset_step:
            if item is ResetStep.OBSERVE and render:
                work += rendering
            if item is ResetStep.EVENTS:
                work.append((True, functools.partial(self._reset_scene, mask)))
                work += [(launch.capturable, functools.partial(launch, reset)) for launch in launches.reset]
            elif item is ResetStep.COMMIT:
                work.append((True, functools.partial(self._commit, mask)))
            elif item is ResetStep.RENDER:
                work += rendering
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
        # Actions apply before every physics step, as in MDP1, so targets relative to the joint state follow it,
        # unless they apply once per control step.
        mdp2.process_actions(self.mdp, self.state, self.actions)
        for _ in range(self.cfg.decimation // self._substeps):
            mdp2.apply_actions(self.mdp, self.state, self.actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self._update_scene(self.physics_dt * self._substeps)

    def _update_scene(self, dt: float) -> None:
        """Refresh the asset data and the sensors other than cameras after a physics step, as the scene would."""
        self.sim.render_context.update_scene_state(self.sim.get_physics_step_count())
        for asset in self._assets:
            asset.update(dt)
        if self._batched_sensors:
            SensorBase.update_batch(self._batched_sensors, dt)
        for sensor in self._sensors:
            sensor.update(dt, force_recompute=True)

    def _render(self) -> None:
        """Render every camera in every world. Part of the captured step."""
        if not self._cameras:
            return
        if all(camera.supports_batch_update for camera in self._cameras):
            SensorBase.update_batch(self._cameras, self.step_dt)
            return
        for camera in self._cameras:
            camera.update(self.step_dt, force_recompute=True)

    def _views(self, buffers: dict[str, ProxyArray]) -> VecEnvObs:
        """Return Torch views of observation buffers, grouped: a tensor per group, or a dict of term tensors."""
        observations: VecEnvObs = {}
        for buffer, values in buffers.items():
            if "/" not in buffer:
                observations[buffer] = values.torch
                continue
            group, term = buffer.split("/")
            shape = self.mdp.observation_layouts[buffer][0]
            observations.setdefault(group, {})[term] = values.torch.view(self.num_envs, *shape)
        return observations

    def _commit(self, mask: ProxyArray) -> None:
        """Commit the writes of the reset events in the masked worlds, and of the interval events of the last step in
        the worlds where they ran. Part of the captured step."""
        if self._event_worlds is None:
            self.scene_data.commit(mask.warp, self.mdp.reset_writes)
            return
        torch.logical_or(mask.torch, self._event_worlds.torch, out=self._commit_mask.torch)
        self.scene_data.commit(self._commit_mask.warp, self.mdp.reset_writes | self.mdp.step_writes)

    def _reset_scene(self, mask: ProxyArray) -> None:
        """Reset the assets and sensors of the masked worlds, before the MDP's reset events."""
        assets = [*self.scene.articulations.values(), *self.scene.rigid_objects.values()]
        for asset in assets:
            asset.reset(env_mask=mask.warp)
        # Clear sensor histories, so that a new episode does not see the contacts that ended the last one.
        for sensor in self.scene.sensors.values():
            sensor.reset(env_mask=mask.warp)


def _without_renderer_graphs(scene_cfg: Any) -> dict[str, CameraCfg]:
    """Return the scene's Newton cameras with renderers that do not capture graphs of their own.

    The step's CUDA graph already holds the renders. With several cameras, the graph that the renderer captures per
    camera goes stale once the scene allocates more memory, and renders black images.
    """
    return {
        name: dataclasses.replace(camera, renderer_cfg=dataclasses.replace(camera.renderer_cfg, use_cuda_graph=False))
        for name, camera in scene_cfg.__dict__.items()
        if isinstance(camera, CameraCfg) and isinstance(camera.renderer_cfg, NewtonWarpRendererCfg)
    }


_OPTIONAL_STEPS = (Step.RENDER, ResetStep.RENDER)  # Built-in work that a step may leave out


def _check_step(cfg: MDP2EnvCfg) -> None:
    """Raise if the config's step or reset step leaves out, repeats, or reorders its built-in work. Rendering may be
    left out."""
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
        required = [item for item in order if item not in _OPTIONAL_STEPS]
        if builtin != [item for item in order if item in builtin] or not set(required) <= set(builtin):
            optional = [str(item) for item in order if item in _OPTIONAL_STEPS]
            raise ValueError(
                f"MDP2EnvCfg.{name} must hold {', '.join(map(str, order))} once each, in this order, with stages"
                f" anywhere between them and {', '.join(optional)} optional, got {[str(item) for item in builtin]}:"
                f" {reason}."
            )


class MDP2RLEnv(MDP2Env, gym.Env):
    """A Gym environment driven by an MDP2 task, like a :class:`~isaaclab.envs.ManagerBasedRLEnv` without managers.

    :meth:`step` returns copies of the observations after resets, and the rewards and the terminated and truncated
    flags of the step as Torch views of buffers that the next step overwrites. A world is truncated only if it did not
    terminate. ``extras["log"]`` holds, per term, the average over worlds of their last finished episode: the reward
    sum per second, whether the term ended it, and the value that each recorded quantity had at its end. With
    :attr:`~MDP2RLEnvCfg.compute_final_obs`, ``extras["final_obs"]`` holds the observations that the worlds that
    ended had before their reset, also as views; only the rows of the worlds that ended are of this step.
    """

    is_vector_env: ClassVar[bool] = True
    diagnostics: ClassVar[bool] = True
    metadata: ClassVar[dict[str, Any]] = {"render_modes": [None]}

    cfg: MDP2RLEnvCfg

    def __init__(self, cfg: MDP2RLEnvCfg, render_mode: str | None = None, **kwargs: Any) -> None:
        self.final_observations = cfg.compute_final_obs
        self._budgeted = cfg.episode_budget is not None
        super().__init__(cfg)
        self.render_mode = render_mode
        self.metadata = {**self.metadata, "render_fps": 1 / self.step_dt}
        self.obs_buf: VecEnvObs = {}
        self._bind()

    def _bind(self) -> None:
        """Bind the spaces, the final observations, and the episode statistics to the MDP."""
        cfg = self.cfg
        if cfg.compute_final_obs:
            self.extras["final_obs"] = self._views(self.transition.final_observations)
        spaces: dict[str, Any] = {}
        for buffer, (shape, dtype) in self.mdp.observation_layouts.items():
            if dtype is wp.float32:
                space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape)
            else:
                info = np.iinfo(wp.dtype_to_numpy(dtype))
                space = gym.spaces.Box(low=info.min, high=info.max, shape=shape, dtype=wp.dtype_to_numpy(dtype))
            if "/" in buffer:
                group, term = buffer.split("/")
                spaces.setdefault(group, {})[term] = space
            else:
                spaces[buffer] = space
        self.single_observation_space = gym.spaces.Dict(
            {group: gym.spaces.Dict(space) if isinstance(space, dict) else space for group, space in spaces.items()}
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
            if op.record
        }
        # The time limit is not an op: whether it ended the last episode of each world is kept apart, under its name.
        time_limits = [name for name, term in _terminations(cfg.task) if isinstance(term, mdp2.TimeLimit)]
        self._time_limit = time_limits[0] if time_limits else None
        # Progress: whether each task with a completion was done in each world's last finished episode, logged over
        # the worlds of the world configs that have it, and for a task of a choice over those that drew it.
        paths = list(dict.fromkeys(path for tree in self.mdp.tasks for path in sorted(tree.completes)))
        drawn = {node.path for tree in self.mdp.tasks for node in tree.nodes if tree.drawn(node)}
        everywhere = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._task_runs = [
            self.state.arrays[mdp2.progress_field(path, "active")].torch if path in drawn else everywhere
            for path in paths
        ]
        self._last_runs = torch.ones(self.num_envs, len(paths), device=self.device)
        self._tasks = {
            f"Progress/{path or 'task'}": self.state.arrays[mdp2.progress_field(path, "done")].torch for path in paths
        }
        self._task_worlds = []
        for path in paths:
            worlds = [world for tree in self.mdp.tasks if path in tree.completes for world in tree.worlds]
            self._task_worlds.append(slice(None) if len(worlds) == self.num_envs else worlds)
        self._last_done = torch.zeros(self.num_envs, len(self._tasks), device=self.device)
        self._last_time_outs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if cfg.episode_budget is not None:
            self._init_episode_budget(cfg.episode_budget)

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
        if self.cfg.episode_budget is not None:
            self._restart_episode_budget()

        # Learners keep the observations across the next step, which overwrites the buffers.
        self.obs_buf = _clone(self.observations)
        return self.obs_buf, self.extras

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Apply the actions, step the physics, evaluate the MDP, reset the worlds that ended, and observe."""
        self._step_or_replay(action)
        self._log_episodes()
        self.obs_buf = _clone(self.observations)

        transition = self.transition
        return (
            self.obs_buf,
            transition.rewards.torch,
            transition.terminated.torch,
            transition.truncated.torch,
            self.extras,
        )

    @property
    def episodes_completed(self) -> int:
        """Episodes that ended since the last :meth:`reset`, with an episode budget. Waits for the device."""
        return int(self._completed.item())

    def episode_records(self) -> dict[str, torch.Tensor]:
        """Return a row per completed episode, with an episode budget, as columns by name, in the order they ended.

        The columns are ``world``, ``episode`` (its index among the world's episodes), ``length`` (control steps),
        ``terminated`` and ``truncated``, the value at the end of each recorded quantity, whether each termination
        ended it, as ``termination/<name>``, and, for each task with a completion, the episode step at which it was
        done (-1 if it was not) as ``progress/<path>``, the task's score as ``progress/score``, how often each task that
        repeats was done as ``progress/<path>/count``, and, for a task with tunable values, the tunings its world had
        applied as ``revision``. Rows of episodes that ended in the same
        step are ordered by world. Waits for the device.
        """
        rows = self._records[: self.episodes_completed]
        return {name: rows[:, k] for k, name in enumerate(self._record_columns)}

    def _init_episode_budget(self, budget: int) -> None:
        """Allocate the episode counters and the record buffer. The buffer has a spare last row, which the worlds
        that did not end write in the captured step."""
        device = self.device
        self._budget = budget
        self._started = torch.zeros((), dtype=torch.long, device=device)
        self._completed = torch.zeros((), dtype=torch.long, device=device)
        self._active = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self._ended = torch.zeros(self.num_envs, dtype=torch.bool, device=device)  # Episodes that ended this step
        self._episode_index = torch.zeros(self.num_envs, device=device)
        recorded = [op for op in self.mdp.ops[mdp2.Phase.QUANTITIES] if op.record]
        self._recorded = [self.state.arrays[op.writes[0]].torch for op in recorded]
        self._num_quantities = len(recorded)
        self._record_columns = ["world", "episode", "length", "terminated", "truncated"]
        self._record_columns += [op.name.split(".", 1)[1] for op in recorded]
        self._record_columns += [f"termination/{name}" for name in self._termination_names]
        completes = list(dict.fromkeys(path for tree in self.mdp.tasks for path in sorted(tree.completes)))
        self._recorded += [self.state.arrays[mdp2.progress_field(path, "done_at")].torch for path in completes]
        self._record_columns += [f"progress/{path or 'task'}" for path in completes]
        if completes:
            self._recorded.append(self.state.arrays[mdp2.progress_field("", "score")].torch)
            self._record_columns.append("progress/score")
        repeats = list(dict.fromkeys(node.path for tree in self.mdp.tasks for node in tree.nodes if node.repeat))
        self._recorded += [self.state.arrays[mdp2.progress_field(path, "count")].torch for path in repeats]
        self._record_columns += [f"progress/{path or 'task'}/count" for path in repeats]
        if self.state.revision is not None:
            self._recorded.append(self.state.revision.torch)
            self._record_columns.append("revision")
        self._records = torch.zeros(budget + 1, len(self._record_columns), device=device)
        self._worlds = torch.arange(self.num_envs, device=device)

    def _restart_episode_budget(self) -> None:
        """Start the first episodes of the budget, in the first worlds."""
        self._started.fill_(min(self._budget, self.num_envs))
        self._completed.zero_()
        self._episode_index.zero_()
        torch.lt(self._worlds, self._budget, out=self._active)

    def _spend_episode_budget(self) -> None:
        """Record the episodes of active worlds that ended, and restart only as many as the budget has left.

        Worlds that do not restart become inactive: they keep stepping, but their episodes no longer end or reset.
        ``done`` stays whether an active world's episode ended; the worlds to reset are those that restart. Part of the
        captured step, so it works on device tensors only.
        """
        transition, ended = self.transition, self._ended
        torch.logical_and(transition.done.torch, self._active, out=ended)
        rank = torch.cumsum(ended, dim=0)  # 1 for the first world that ended, and so on

        # Rows of the ended episodes, in world order after those of earlier steps; the others write the spare row.
        rows = torch.where(ended, self._completed + rank - 1, self._budget)
        values = [
            self._worlds.float(),
            self._episode_index,
            self.state.episode_length.torch.float(),
            transition.terminated.torch.float(),
            transition.truncated.torch.float(),
            *self._recorded[: self._num_quantities],
            *transition.termination_terms.torch.float().unbind(1),
            *(values.float() for values in self._recorded[self._num_quantities :]),
        ]
        self._records.index_copy_(0, rows, torch.stack(values, dim=1))
        self._completed.add_(rank[-1])
        self._episode_index.add_(ended.float())

        restarted = ended & (self._started + rank <= self._budget)
        self._started.add_(restarted.sum())
        self._active.logical_and_(~ended | restarted)
        # Only the worlds that restart are reset; the others report their episode once.
        reported = ended | self._active
        transition.terminated.torch.logical_and_(reported)
        transition.truncated.torch.logical_and_(reported)
        transition.done.torch.logical_and_(reported)
        self._resets.torch.copy_(restarted)

    def _after_evaluate(self) -> None:
        """Keep the recorded quantities of the worlds that ended, and whether they timed out, before their reset."""
        # With an episode budget, only the episodes of active worlds count, and only some of them restart.
        if self.cfg.episode_budget is not None:
            self._spend_episode_budget()
        done = self._ended if self.cfg.episode_budget is not None else self.transition.done.torch
        for value, last, _ in self._metrics.values():
            torch.where(done, value, last, out=last)
        if self._tasks:
            done_tasks = torch.stack(list(self._tasks.values()), 1).float()
            torch.where(done[:, None], done_tasks, self._last_done, out=self._last_done)
            runs = torch.stack(self._task_runs, 1).float()
            torch.where(done[:, None], runs, self._last_runs, out=self._last_runs)
        if self._time_limit is not None:
            timed_out = self.state.episode_length.torch >= self.max_episode_length
            torch.where(done, timed_out, self._last_time_outs, out=self._last_time_outs)

    def _log_episodes(self) -> None:
        """Add the step's term rewards to the episode sums, and record the episodes that ended in ``extras["log"]``."""
        transition = self.transition
        ended = (self._ended if self.cfg.episode_budget is not None else transition.done.torch)[:, None]
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
        # A task is averaged over the worlds of its world config.
        for k, (name, worlds) in enumerate(zip(self._tasks, self._task_worlds)):
            runs = self._last_runs[worlds, k]
            log[name] = (self._last_done[worlds, k] * runs).sum() / runs.sum().clamp(min=1.0)
        self.extras["log"] = log


def _terminations(task: mdp2.TaskCfg) -> list[tuple[str, Any]]:
    """Return the terminations of a task and of its subtasks, by name."""
    subtasks = [_terminations(subtask) for subtask in task.subtasks.values()]
    return [*task.terminations.items(), *(item for terminations in subtasks for item in terminations)]


def _clone(observations: VecEnvObs) -> VecEnvObs:
    """Copy observations, also the term tensors of groups that are not concatenated."""
    return {
        group: {term: values.clone() for term, values in value.items()} if isinstance(value, dict) else value.clone()
        for group, value in observations.items()
    }
