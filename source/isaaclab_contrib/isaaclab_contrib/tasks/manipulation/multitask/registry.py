# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MultiTaskRegistry: composes robot and task modules into a full RL environment config.

Usage example::

    from isaaclab_contrib.tasks.manipulation.multitask.registry import MultiTaskRegistry
    from isaaclab_contrib.tasks.manipulation.multitask.robots import FRANKA_ROBOT, OPENARM_ROBOT, UR10_ROBOT
    from isaaclab_contrib.tasks.manipulation.multitask.tasks import (
        LIFT_TASK_OPENARM,
        CABINET_TASK,
        REACH_TASK,
    )

    env_cfg = (
        MultiTaskRegistry()
        .register(OPENARM_ROBOT, LIFT_TASK_OPENARM)
        .register(FRANKA_ROBOT, CABINET_TASK)
        .register(UR10_ROBOT, REACH_TASK)
        .build_env_cfg(num_envs=4096)
    )

The registry assembles:

* **Scene** — all robot and task assets in per-group
  :class:`~isaaclab.cloner.InclusionSet` clone combinations.
* **Actions** — groups action specs by column key into
  :class:`~...mdp.actions_cfg.ScatteredActionTermCfg` entries.
* **Commands** — merges command cfgs from all task modules.
* **Observations** — merges scatter obs from all robots/tasks into shared
  ``scatter_term`` slots; adds group-local task obs, ``task_onehot``, and
  ``last_action``.
* **Rewards** — merges per-task rewards; adds global ``action_rate`` and
  ``joint_vel`` penalties.
* **Terminations** — ``time_out`` plus per-task conditions.
* **Events** — per-robot and per-task reset terms.
* **Curriculum** — default action-rate and joint-velocity weight schedules.
"""

from __future__ import annotations

from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import CloneCfg, InclusionSet
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBaseCfg as TermCfg
from isaaclab.managers import ObservationGroupCfg, SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg, SelectorCfg, SelectorTermCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.tasks.manipulation.multitask import mdp, selectors

from .robots._base import RobotModuleCfg
from .tasks._base import TaskModuleCfg

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _DynCfg:
    """Minimal config namespace.

    IsaacLab's managers iterate ``cfg.__dict__`` to collect terms, so any
    object with the right attributes works as a manager config.  This class
    provides a lightweight alternative to a full ``@configclass`` for
    dynamically assembled manager configs (actions, rewards, events, etc.).
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.__dict__})"


# ---------------------------------------------------------------------------
# Registration record
# ---------------------------------------------------------------------------


@dataclass
class Registration:
    """A single (robot, task) entry in the registry."""

    robot: RobotModuleCfg
    task: TaskModuleCfg
    group_name: str | None = None
    weight: int = 1

    @property
    def group(self) -> str:
        """Resolved clone-group name (defaults to ``f"{robot.name}_{task.name}"``)."""
        return self.group_name or f"{self.robot.name}_{self.task.name}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class MultiTaskRegistry:
    """Assembles a :class:`~isaaclab.envs.ManagerBasedRLEnvCfg` from
    robot-task module pairs.

    Call :meth:`register` for each (robot, task) combination, then
    :meth:`build_env_cfg` to obtain a fully assembled environment config.
    The registry returns ``self`` from :meth:`register` to support method
    chaining.
    """

    def __init__(self) -> None:
        self._regs: list[Registration] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        robot: RobotModuleCfg,
        task: TaskModuleCfg,
        *,
        group_name: str | None = None,
        weight: int = 1,
    ) -> MultiTaskRegistry:
        """Add a (robot, task) pair to the registry.

        Args:
            robot: Robot module instance.
            task: Task module instance.
            group_name: Override the default clone-group name
                ``f"{robot.name}_{task.name}"`` (e.g. ``"franka_lift"``).
            weight: Relative number of environments allocated to this group.
                The actual env count is proportional to this weight.

        Returns:
            ``self`` for method chaining.
        """
        self._regs.append(Registration(robot, task, group_name, weight))
        return self

    def build_env_cfg(
        self,
        num_envs: int = 4096,
        env_spacing: float = 2.5,
        *,
        replicate_physics: bool = True,
        physics=None,
        decimation: int = 2,
        episode_length_s: float = 8.0,
        sim_dt: float = 1.0 / 60.0,
        action_rate_curriculum_steps: int = 100_000,
        joint_vel_curriculum_steps: int = 100_000,
    ) -> ManagerBasedRLEnvCfg:
        """Build and return a complete :class:`~isaaclab.envs.ManagerBasedRLEnvCfg`.

        Args:
            num_envs: Total number of parallel environments.
            env_spacing: Distance between environment origins [m].
            replicate_physics: If ``True``, physics is replicated across envs.
            physics: Optional physics backend config
                (e.g. :class:`~...config.demo.demo_multi_robot_reach_env_cfg.MultitaskPhysicsCfg`).
                When ``None``, the simulator default is used.
            decimation: Number of physics steps per policy step.
            episode_length_s: Maximum episode duration [s].
            sim_dt: Physics simulation timestep [s].
            action_rate_curriculum_steps: Step count at which the
                ``action_rate`` penalty reaches its target weight.
            joint_vel_curriculum_steps: Step count at which the
                ``joint_vel`` penalty reaches its target weight.

        Returns:
            Fully assembled environment config ready to pass to
            :class:`~isaaclab.envs.ManagerBasedRLEnv`.
        """
        if not self._regs:
            raise RuntimeError("No robot-task pairs registered. Call register() first.")

        scene = self._build_scene(num_envs, env_spacing, replicate_physics)
        actions = self._build_actions()
        commands = self._build_commands()
        observations = self._build_observations()
        rewards = self._build_rewards()
        terminations = self._build_terminations()
        events = self._build_events()
        curriculum = self._build_curriculum(action_rate_curriculum_steps, joint_vel_curriculum_steps)

        # ── Capture sim params in closure for __post_init__ ───────────────
        _decimation = decimation
        _episode_length_s = episode_length_s
        _sim_dt = sim_dt
        _physics = physics

        def _registry_post_init(self):
            self.decimation = _decimation
            self.episode_length_s = _episode_length_s
            self.sim.dt = _sim_dt
            self.sim.render_interval = _decimation
            if _physics is not None:
                self.sim.physics = _physics

        # ── Create env configclass dynamically ────────────────────────────
        # Returns the *class* (not an instance) so that gym's
        # ``env_cfg_entry_point`` resolver (which calls inspect.isclass())
        # can accept it directly.
        _EnvCls = configclass(
            type(
                "_RegistryEnvCfg",
                (ManagerBasedRLEnvCfg,),
                {
                    "scene": scene,
                    "actions": actions,
                    "commands": commands,
                    "observations": observations,
                    "rewards": rewards,
                    "terminations": terminations,
                    "events": events,
                    "curriculum": curriculum,
                    "__post_init__": _registry_post_init,
                },
            )
        )
        return _EnvCls

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_scene(
        self,
        num_envs: int,
        env_spacing: float,
        replicate_physics: bool,
    ) -> InteractiveSceneCfg:
        """Build scene config with all robot and task assets."""

        # ── Detect shared robot assets ────────────────────────────────
        # A robot-asset key that appears in more than one registration is
        # "global" (the physical robot is shared across all environments).
        # Global assets must NOT appear in any clone group; only task-local
        # objects go into the clone group's asset list.
        # Examples:
        #   Multi-robot multi-task → each robot key is unique → all in groups.
        #   Single-robot multi-task → "franka_robot" appears N times → global.
        robot_key_count: dict[str, int] = {}
        for reg in self._regs:
            for key in reg.robot.scene_assets(reg.group):
                robot_key_count[key] = robot_key_count.get(key, 0) + 1
        shared_robot_keys = {k for k, n in robot_key_count.items() if n > 1}

        # ── Clone combinations and selectors ─────────────────────────
        clone_combinations: list[InclusionSet] = []
        selector_terms: dict[str, SelectorTermCfg] = {}
        for reg in self._regs:
            # Exclude shared robot keys from clone combinations so they remain
            # global, but use task-local assets to select each registration's envs.
            robot_keys = [k for k in reg.robot.scene_assets(reg.group) if k not in shared_robot_keys]
            task_keys = list(reg.task.scene_assets(reg.group, reg.robot).keys())
            clone_asset_names = robot_keys + task_keys
            clone_combinations.append(InclusionSet(assets=clone_asset_names, weight=reg.weight))
            selector_terms[reg.group] = SelectorTermCfg(func=selectors.asset_names, params={"names": clone_asset_names})

        _SelectorCfg = configclass(type("_RegistrySelectorCfg", (SelectorCfg,), selector_terms))

        # ── All scene assets ──────────────────────────────────────────
        scene_attrs: dict[str, object] = {
            "clone_cfg": CloneCfg(clone_combinations=clone_combinations),
            "selector_cfg": _SelectorCfg(),
            "plane": AssetBaseCfg(
                prim_path="/World/GroundPlane",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
                spawn=GroundPlaneCfg(),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
            ),
        }
        for reg in self._regs:
            scene_attrs.update(reg.robot.scene_assets(reg.group))
            scene_attrs.update(reg.task.scene_assets(reg.group, reg.robot))

        # ── Build InteractiveSceneCfg subclass dynamically ────────────
        _SceneCls = configclass(type("_RegistrySceneCfg", (InteractiveSceneCfg,), scene_attrs))
        return _SceneCls(
            num_envs=num_envs,
            env_spacing=env_spacing,
            replicate_physics=replicate_physics,
        )

    def _build_actions(self) -> _DynCfg:
        """Group action specs by column key into ScatteredActionTermCfg entries."""

        # {column_key: (dim, [ActionTermCfg, ...])}
        columns: dict[str, tuple[int, list]] = {}
        for reg in self._regs:
            for key, (dim, spec) in reg.robot.action_specs(reg.group).items():
                if key not in columns:
                    columns[key] = (dim, [])
                columns[key][1].append(spec)

        actions_dict: dict[str, object] = {}
        for key, (dim, specs) in columns.items():
            actions_dict[key] = mdp.ScatteredActionTermCfg(dim=dim, terms=specs)
        return _DynCfg(**actions_dict)

    def _build_commands(self) -> _DynCfg:
        """Merge command configs from all task modules."""
        cmds: dict[str, object] = {}
        for reg in self._regs:
            cmds.update(reg.task.command_terms(reg.group, reg.robot))
        return _DynCfg(**cmds)

    def _build_observations(self) -> _DynCfg:
        """Build the observations config.

        * **Global terms**: ``task_onehot``, ``actions``.
        * **Scatter slots**: merged from robot and task scatter contributions
          (e.g. ``ee_pose``, ``commands``, ``ee_pos_error``).
        * **Task-local terms**: group-specific obs namespaced as
          ``f"{group}_{local_name}"``.
        """

        # ── Collect scatter contributions ─────────────────────────────
        # {slot: [(output_dim, inner_TermCfg), ...]}
        scatter_slots: dict[str, list[tuple[int | None, TermCfg]]] = {}
        for reg in self._regs:
            for slot, (dim, term) in reg.robot.scatter_obs_terms(reg.group).items():
                scatter_slots.setdefault(slot, []).append((dim, term))
            for slot, (dim, term) in reg.task.scatter_obs_terms(reg.group, reg.robot).items():
                scatter_slots.setdefault(slot, []).append((dim, term))

        # ── Build policy obs dict ─────────────────────────────────────
        policy_obs: dict[str, ObsTerm] = {}
        policy_obs["task_onehot"] = ObsTerm(func=mdp.multi_task_onehot)

        for slot, contributions in scatter_slots.items():
            dims, terms = zip(*contributions)
            output_dim = next((d for d in dims if d is not None), None)
            params: dict[str, object] = {"terms": list(terms)}
            if output_dim is not None:
                params["output_dim"] = output_dim
            policy_obs[slot] = ObsTerm(func=mdp.scatter_term, params=params)

        # ── Task-local terms ──────────────────────────────────────────
        for reg in self._regs:
            for local_name, term in reg.task.task_obs_terms(reg.group, reg.robot).items():
                policy_obs[f"{reg.group}_{local_name}"] = term

        policy_obs["actions"] = ObsTerm(func=mdp.last_action)

        # ── Build PolicyCfg instance ──────────────────────────────────
        # ObservationGroupCfg is a configclass; setattr() adds to __dict__
        # which the ObservationManager iterates via cfg.__dict__.
        policy_cfg = ObservationGroupCfg()
        policy_cfg.enable_corruption = True
        policy_cfg.concatenate_terms = True
        for name, term in policy_obs.items():
            setattr(policy_cfg, name, term)

        return _DynCfg(policy=policy_cfg)

    def _build_rewards(self) -> _DynCfg:
        """Merge task rewards and add global penalties."""

        rewards_dict: dict[str, RewTerm] = {}
        for reg in self._regs:
            rewards_dict.update(reg.task.reward_terms(reg.group, reg.robot))

        # ── Global action-rate penalty ────────────────────────────────
        rewards_dict["action_rate"] = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

        # ── Global joint-velocity penalty (scatter across all robots) ─
        jv_terms: list[TermCfg] = [
            TermCfg(
                func=mdp.joint_vel_l2,
                params={
                    "asset_cfg": SceneEntityCfg(
                        f"{reg.robot.name}_robot",
                        joint_names=reg.robot.all_joint_names,
                        selector=reg.group,
                    )
                },
            )
            for reg in self._regs
        ]
        rewards_dict["joint_vel"] = RewTerm(
            func=mdp.scatter_term,
            weight=-1e-4,
            params={"output_dim": 0, "terms": jv_terms},
        )

        return _DynCfg(**rewards_dict)

    def _build_terminations(self) -> _DynCfg:
        """Merge task terminations and add global time_out."""
        terms: dict[str, DoneTerm] = {"time_out": DoneTerm(func=mdp.time_out, time_out=True)}
        for reg in self._regs:
            terms.update(reg.task.termination_terms(reg.group, reg.robot))
        return _DynCfg(**terms)

    def _build_events(self) -> _DynCfg:
        """Merge robot and task reset events."""
        events: dict[str, EventTerm] = {}
        for reg in self._regs:
            events.update(reg.robot.reset_events(reg.group))
            events.update(reg.task.reset_events(reg.group, reg.robot))
        return _DynCfg(**events)

    def _build_curriculum(
        self,
        action_rate_steps: int,
        joint_vel_steps: int,
    ) -> _DynCfg:
        """Default curriculum: ramp up action-rate and joint-vel penalties."""
        curriculum: dict[str, CurrTerm] = {
            "action_rate": CurrTerm(
                func=mdp.modify_reward_weight,
                params={
                    "term_name": "action_rate",
                    "weight": -1e-2,
                    "num_steps": action_rate_steps,
                },
            ),
            "joint_vel": CurrTerm(
                func=mdp.modify_reward_weight,
                params={
                    "term_name": "joint_vel",
                    "weight": -1e-2,
                    "num_steps": joint_vel_steps,
                },
            ),
        }
        return _DynCfg(**curriculum)
