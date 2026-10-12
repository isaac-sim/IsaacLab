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
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
import warp as wp

from ...managers import ManagerBase
from ...utils.step_graph import StepGraph, StepOp

if TYPE_CHECKING:
    from ..direct_rl_env import DirectRLEnv
    from ..manager_based_rl_env import ManagerBasedRLEnv

logger = logging.getLogger(__name__)

# Step outputs that each step binds anew and that no earlier operation of the step reads.
_OUTPUTS = frozenset(f"env.{n}" for n in ("obs_buf", "reward_buf", "reset_buf", "reset_terminated", "reset_time_outs"))


class EnvStepCapture:
    """Run an environment step as a :class:`~isaaclab.utils.StepGraph` around the physics step's operations.

    Consecutive graphable operations replay from one CUDA graph; the rest run eagerly in place. The first launch after a
    (re)build runs eagerly and turns operations that synchronize the device into eager islands; the next launch records
    the graphs and replays them. The step is rebuilt when the physics operations change or :meth:`invalidate` is called,
    and steps eagerly for good when rebuilds recur every few steps or recording fails.
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
            build_ops: Returns the step as operations around the physics operations; they read :attr:`action`.
            blocker: Returns why the step cannot be captured at all, or None.
            enabled: Whether to capture.
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
        if self.disabled_reason is not None or env.sim.is_rendering or env.video_recorders:
            return None
        physics_ops = env.sim.physics_manager.step_ops()
        if physics_ops is None:
            return self.disable("the physics backend exposes no step operations")
        if self.graph is not None:
            if physics_ops is self._physics_ops:
                return self.graph
            self._rebuilds.append(env.common_step_counter)
        # rebuilding every few steps costs more than it saves: some term keeps changing captured constants
        if sum(env.common_step_counter - step < 50 for step in self._rebuilds) > 3:
            return self.disable("the captured step is invalidated every few steps")
        if (reason := self._blocker() or self._scene_blocker()) is not None:
            return self.disable(reason)
        if self.action is None or self.action.shape != action.shape:
            self.action = torch.zeros(action.shape, dtype=torch.float32, device=env.device)
        self._physics_ops = physics_ops
        self.graph = self._graph_from(self._build_ops(physics_ops))
        return self.graph

    def decimation_ops(self, physics_ops: list[StepOp], first: StepOp, writes: list[StepOp]) -> list[StepOp]:
        """Lay out the decimation loop: ``first`` once, then per physics call ``writes``, physics and scene update.

        Physics may open with eager host work (e.g. model-change notification) that ``first`` does not depend on;
        running ``first`` after it lets it share a graph with the physics step.
        """
        env = self._env
        steps_per_call = env.cfg.decimation if env._physics_handles_decimation else 1
        leading = next((i for i, op in enumerate(physics_ops) if op.graphable), len(physics_ops))
        update = StepOp(partial(env.scene.update, dt=env.physics_dt * steps_per_call), True, "scene.update")
        ops = [*physics_ops[:leading], first]
        for i in range(env.cfg.decimation // steps_per_call):
            ops += [*(physics_ops[:leading] if i else ()), *writes, *physics_ops[leading:], update]
        return ops

    def disable(self, reason: str) -> None:
        """Step eagerly from now on, logging ``reason`` once."""
        logger.warning("Stepping %s eagerly: %s.", type(self._env).__name__, reason)
        self.disabled_reason = reason
        self.graph = None

    def launch(self, graph: StepGraph, action: torch.Tensor) -> None:
        """Run one environment step from ``graph``, then the host bookkeeping that replayed operations skip."""
        env = self._env
        self.action.copy_(action)
        if not graph.captured and self._warm is graph:
            # the previous step ran the operations eagerly; record them now and replay, so returned buffers hold
            # executed work
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

    def _scene_blocker(self) -> str | None:
        scene = self._env.scene
        if not scene.cfg.lazy_sensor_update:
            return "the scene updates sensors eagerly (lazy_sensor_update is False)"
        if self._env.has_rtx_sensors:
            return "camera sensors render on the host"
        for name, sensor in scene.sensors.items():
            if not sensor.supports_graph_capture:
                return f"sensor '{name}' ({type(sensor).__name__}) cannot be recorded"
        return None

    def _graph_from(self, ops: list[StepOp]) -> StepGraph:
        """Group operations into a step graph, keeping operations known to synchronize the device eager."""
        return StepGraph([StepOp(op.fn, False, op.name) if op.name in self._sync_ops else op for op in ops])

    def _launch_watching_syncs(self, graph: StepGraph) -> StepGraph:
        """Run the operations eagerly; operations that synchronized the device become eager islands.

        An eager island that rebinds buffers every step would leave recorded readers on stale tensors, so it disables
        capture.
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
                    count, before = len(caught), self._buffers()
                    op.fn()
                    after = self._buffers()
                    rebinds.setdefault(op.name, set()).update(
                        n for n, b in before.items() if n in after and after[n] is not b
                    )
                    if op.graphable and any("synchroniz" in str(w.message) for w in caught[count:]):
                        synced.add(op.name)
            finally:
                torch.cuda.set_sync_debug_mode(previous_mode)
        if synced:
            logger.info("Running %s eagerly in the captured step: they synchronize the device.", sorted(synced))
            self._sync_ops |= synced
            graph = self.graph = self._graph_from(graph.ops)
        for op in graph.ops:
            if not op.graphable and rebinds.get(op.name):
                attributes = ", ".join(sorted(rebinds[op.name]))
                self.disable(f"'{op.name}' runs eagerly and rebinds {attributes}; write them in place to capture")
                break
        return graph

    def _capture_carry(self, before: dict[str, Any], after: dict[str, Any]) -> Any | None:
        """Record copies of rebound buffers into the buffers that operations recorded before the rebinding read.

        A step that replaces a buffer (``self.x = f(self.x)``) leaves earlier recorded readers on the old tensor, so
        each replay ends by copying the new value into it, as eager stepping would hand it to the next step's readers.
        """
        pairs = []
        for name, old in before.items():
            new = after.get(name)
            if new is None or new is old:
                continue
            if type(new) is not type(old) or new.shape != old.shape or new.dtype != old.dtype:
                raise RuntimeError(f"{name} is rebound to a buffer of another shape or type")
            pairs.append((old, new))
        # the recorded copies address these tensors: they must outlive the graph
        self._carry_pairs = pairs
        if not pairs:
            return None
        return self._capture(lambda: [o.copy_(n) if isinstance(o, torch.Tensor) else wp.copy(o, n) for o, n in pairs])

    def _buffers(self) -> dict[str, Any]:
        """Tensors held by the environment, its managers and their terms, by attribute path, except step outputs."""
        found = {}
        for owner_name, owner in self._buffer_owners():
            for name, value in vars(owner).items():
                items = value.items() if isinstance(value, dict) else [(None, value)]
                for key, item in items:
                    path = f"{owner_name}.{name}"
                    if isinstance(item, (torch.Tensor, wp.array)) and path not in _OUTPUTS:
                        found[path if key is None else f"{path}[{key!r}]"] = item
        return found

    def _buffer_owners(self):
        """Yield the environment, its managers and their term objects with readable names."""
        yield "env", self._env
        for manager_name, manager in vars(self._env).items():
            if not isinstance(manager, ManagerBase):
                continue
            yield manager_name, manager
            for name, value in vars(manager).items():
                values = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
                for index, item in enumerate(values):
                    for term in item if isinstance(item, list) else [item]:
                        term = getattr(term, "func", term)
                        if hasattr(term, "__dict__") and not inspect.isroutine(term) and not inspect.isclass(term):
                            yield f"{manager_name}.{name}[{index}]:{type(term).__name__}", term

    def _capture(self, fn: Callable[[], None]) -> Any:
        if (graph := self._env.sim.physics_manager.capture_graph(fn)) is None:
            raise RuntimeError("the physics backend does not capture graphs")
        return graph
