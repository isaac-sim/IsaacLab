# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run an environment step from CUDA graphs, with eager islands for work that cannot be recorded."""

from __future__ import annotations

import inspect
import logging
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import warp as wp

from ...managers import ManagerBase
from ...utils.step_graph import StepGraph, StepOp

if TYPE_CHECKING:
    from ..direct_rl_env import DirectRLEnv
    from ..manager_based_rl_env import ManagerBasedRLEnv

logger = logging.getLogger(__name__)

_REBUILD_WINDOW = 50
"""Environment steps within which repeated rebuilds disable capture."""

_MAX_REBUILDS = 3
"""Rebuilds tolerated within :data:`_REBUILD_WINDOW` steps before stepping eagerly."""


class EnvStepCapture:
    """Run an environment step as a :class:`~isaaclab.utils.StepGraph` around the physics step's operations.

    The environment supplies its step as operations; the physics manager supplies the physics step's operations
    (:meth:`~isaaclab.physics.PhysicsManager.step_ops`). Consecutive graphable operations replay from one CUDA graph and
    the rest run eagerly in place. The first launch after a (re)build runs eagerly, which also performs lazy allocations
    and turns operations that synchronize the device into eager islands; the next launch records the graphs and replays
    them. The step is rebuilt when the physics operations change or :meth:`invalidate` is called, and steps eagerly
    for good when rebuilds recur every few steps or recording fails.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv | DirectRLEnv,
        build_ops: Callable[[list[StepOp]], list[StepOp]],
        blocker: Callable[[], str | None],
        enabled: bool,
    ):
        """Initialize the capture.

        Args:
            env: Environment whose step is captured.
            build_ops: Returns the environment step as operations around the physics step's operations. The operations
                read the action from :attr:`action`.
            blocker: Returns why the step cannot be captured at all, or None. Evaluated when the graph is built.
            enabled: Whether to capture; False always steps eagerly.
        """
        self._env = env
        self._build_ops = build_ops
        self._blocker = blocker
        self.disabled_reason: str | None = None if enabled else "disabled by capture_step"
        """Why the environment steps eagerly for good, or None."""
        self.graph: StepGraph | None = None
        """The step's operations grouped into graphs, or None while stepping eagerly."""
        self.action: torch.Tensor | None = None
        """Stable action buffer the recorded operations read."""
        self._physics_ops: list[StepOp] | None = None
        self._warm: StepGraph | None = None
        self._sync_ops: set[str] = set()
        self._rebuilds: list[int] = []
        self._carry: Any | None = None
        self._carry_pairs: list[tuple[Any, Any]] = []

    def invalidate(self) -> None:
        """Rebuild and record the step again before the next step."""
        if self.graph is not None:
            self.graph = None
            self._rebuilds.append(self._env.common_step_counter)

    def prepare(self, action: torch.Tensor) -> StepGraph | None:
        """Return the step graph for this step, or None to step eagerly.

        Args:
            action: The step's actions, used to size :attr:`action`.
        """
        env = self._env
        if self.disabled_reason is not None:
            return None
        if env.sim.is_rendering or env.video_recorders:
            # rendering and video run on the host on their own cadence
            return None
        physics_ops = env.sim.physics_manager.step_ops()
        if physics_ops is None:
            self.disable("the physics backend exposes no step operations")
            return None
        if self.graph is not None and physics_ops is self._physics_ops:
            return self.graph
        if self.graph is not None:
            self._rebuilds.append(env.common_step_counter)
        recent = [step for step in self._rebuilds if env.common_step_counter - step < _REBUILD_WINDOW]
        if len(recent) > _MAX_REBUILDS:
            # rebuilding every few steps costs more than it saves: some term keeps changing captured constants
            self.disable("the captured step is invalidated every few steps")
            return None
        reason = self._blocker()
        if reason is not None:
            self.disable(reason)
            return None
        if self.action is None or self.action.shape != action.shape:
            self.action = torch.zeros(action.shape, dtype=torch.float32, device=env.device)
        self._physics_ops = physics_ops
        self.graph = self._graph_from(self._build_ops(physics_ops))
        return self.graph

    def disable(self, reason: str) -> None:
        """Step eagerly from now on.

        Args:
            reason: Why, logged once.
        """
        logger.warning("Stepping %s eagerly: %s.", type(self._env).__name__, reason)
        self.disabled_reason = reason
        self.graph = None

    def launch(self, graph: StepGraph, action: torch.Tensor) -> None:
        """Run one environment step from ``graph``, then the host bookkeeping that replayed operations skip.

        Args:
            graph: Graph from :meth:`prepare`.
            action: The step's actions.
        """
        env = self._env
        self.action.copy_(action)
        if not graph.captured and self._warm is graph:
            # The previous step ran the operations eagerly; record them now and replay, so that every returned buffer
            # holds executed work.
            try:
                bound = self._buffers()
                graph.capture(self._capture)
                self._carry = self._capture_carry(bound, self._buffers())
            except Exception as error:  # noqa: BLE001
                graph.graphs = None
                self.disable(f"recording the step failed ({error})")
        replayed = graph.captured
        if replayed:
            graph.launch()
            if self._carry is not None:
                self._carry.launch()
        else:
            graph = self._launch_watching_syncs(graph)
        self._warm = graph
        steps_per_call = env.cfg.decimation if env._physics_handles_decimation else 1
        for _ in range(env.cfg.decimation // steps_per_call):
            env._sim_step_counter += steps_per_call
            env.sim.finish_physics_step()
            if replayed:
                env.scene.notify_update_replayed(dt=env.physics_dt * steps_per_call)
        env.common_step_counter += 1

    def _graph_from(self, ops: list[StepOp]) -> StepGraph:
        """Group operations into a step graph, keeping operations known to synchronize the device eager."""
        sync = self._sync_ops
        return StepGraph([StepOp(op.fn, False, op.name) if op.graphable and op.name in sync else op for op in ops])

    def _launch_watching_syncs(self, graph: StepGraph) -> StepGraph:
        """Run the operations eagerly and return the graph with operations that synchronized the device made eager.

        A synchronizing operation (e.g. a term that selects environments by index) would fail inside a capture, so it
        becomes an eager island instead of preventing the capture.
        """
        if "cuda" not in str(self._env.device):
            graph.launch()
            return graph
        synced: set[str] = set()
        rebinds: dict[str, set[str]] = {}
        previous_mode = torch.cuda.get_sync_debug_mode()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                for op in graph.ops:
                    count, before = len(caught), self._buffer_ids()
                    op.fn()
                    after = self._buffer_ids()
                    rebound = {name for name in before.keys() & after.keys() if before[name] != after[name]}
                    rebound = {name for name in rebound if name.split("[")[0] not in self._outputs}
                    if rebound:
                        rebinds.setdefault(op.name, set()).update(rebound)
                    if op.graphable and any("synchroniz" in str(w.message) for w in caught[count:]):
                        synced.add(op.name)
            finally:
                torch.cuda.set_sync_debug_mode(previous_mode)
        if synced:
            logger.info("Running %s eagerly in the captured step: they synchronize the device.", sorted(synced))
            self._sync_ops |= synced
            graph = self.graph = self._graph_from(graph.ops)
        # Recorded operations keep reading the buffers bound when they were recorded. An eager operation that binds new
        # tensors every step leaves the recorded readers on stale ones.
        for op in graph.ops:
            if not op.graphable and op.name in rebinds:
                attributes = ", ".join(sorted(rebinds[op.name]))
                self.disable(f"'{op.name}' runs eagerly and rebinds {attributes}; write them in place to capture")
                return graph
        return graph

    def _capture_carry(self, before: dict[str, Any], after: dict[str, Any]) -> Any | None:
        """Record copies that carry rebound buffers into the buffers that recorded readers hold.

        Recording binds a buffer that the step replaces (e.g. ``self.x = f(self.x)``) to a new tensor. Operations
        recorded before the replacement keep reading the old tensor, so each replay ends by copying the new value into
        it, as eager stepping would hand the new tensor to the next step's readers.

        Raises:
            RuntimeError: If a rebound buffer changes shape or type, so its value cannot be carried.
        """
        pairs = []
        for name, old in before.items():
            new = after.get(name)
            if new is None or new is old or name.split("[")[0] in self._outputs:
                continue
            if type(new) is not type(old) or new.shape != old.shape or new.dtype != old.dtype:
                raise RuntimeError(f"{name} is rebound to a buffer of another shape or type")
            pairs.append((old, new))
        # the recorded copies address these tensors: they must outlive the graph
        self._carry_pairs = pairs
        if not pairs:
            return None

        def carry():
            for old, new in pairs:
                if isinstance(old, torch.Tensor):
                    old.copy_(new)
                else:
                    wp.copy(old, new)

        return self._capture(carry)

    def _buffer_ids(self) -> dict[str, int]:
        """Identities of the tensors held by the environment, its managers, and their terms."""
        return {name: id(value) for name, value in self._buffers().items()}

    def _buffers(self) -> dict[str, Any]:
        """Tensors held by the environment, its managers, and their terms, by attribute path."""
        buffers = (torch.Tensor, wp.array)
        found = {}
        for owner_name, owner in self._buffer_owners():
            for name, value in vars(owner).items():
                if isinstance(value, buffers):
                    found[f"{owner_name}.{name}"] = value
                elif isinstance(value, dict):
                    for key, item in value.items():
                        if isinstance(item, buffers):
                            found[f"{owner_name}.{name}[{key!r}]"] = item
        return found

    def _buffer_owners(self):
        """Yield the environment, its managers, and their term objects with readable names."""
        env = self._env
        yield "env", env
        for manager_name, manager in vars(env).items():
            if not isinstance(manager, ManagerBase):
                continue
            yield manager_name, manager
            for name, value in vars(manager).items():
                values = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
                for index, item in enumerate(values):
                    items = item if isinstance(item, list) else [item]
                    for term in items:
                        term = getattr(term, "func", term)
                        if hasattr(term, "__dict__") and not inspect.isroutine(term) and not inspect.isclass(term):
                            yield f"{manager_name}.{name}[{index}]:{type(term).__name__}", term

    _outputs = frozenset(
        f"env.{name}" for name in ("obs_buf", "reward_buf", "reset_buf", "reset_terminated", "reset_time_outs")
    )
    """Step outputs that each step binds anew and that no earlier operation of the step reads."""

    def _capture(self, fn: Callable[[], None]) -> Any:
        graph = self._env.sim.physics_manager.capture_graph(fn)
        if graph is None:
            raise RuntimeError("the physics backend does not capture graphs")
        return graph
