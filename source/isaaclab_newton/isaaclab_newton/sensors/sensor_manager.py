# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton sensor scheduling and graph execution."""

from __future__ import annotations

import logging
from collections.abc import Callable

import newton
import numpy as np
import warp as wp

logger = logging.getLogger(__name__)


class NewtonSensorManager:
    """Manage Newton sensor updates against a shared model and state.

    Sensors register graph-capturable update functions under unique names. The
    manager owns the shape-BVH refit required by scene-query sensors and captures
    all registered updates into one conditional CUDA graph. Each call to
    :meth:`update` enables only the requested sensors, while the BVH is refit at
    most once after the simulation state changes.

    The Newton model and state are injected instead of read from
    :class:`~isaaclab_newton.physics.NewtonManager`. This keeps sensor
    implementations independent of the physics manager's storage layout. Use
    :meth:`set_state` whenever a solver swaps its active state object.
    """

    def __init__(
        self,
        model: newton.Model,
        state: newton.State,
        device: str,
        use_cuda_graph: bool,
        capture_fn: Callable[[Callable[[], None]], wp.Graph | None],
    ):
        """Initialize the sensor manager.

        Args:
            model: Newton model queried by managed sensors.
            state: Current Newton simulation state.
            device: Device on which sensor updates execute.
            use_cuda_graph: Whether to capture sensor updates into a CUDA graph.
            capture_fn: Function that captures a graph-capturable callable. It
                must return ``None`` when capture is unavailable or fails.
        """
        self._model = model
        self._state = state
        self._device = device
        self._capture_fn = capture_fn
        self._use_cuda_graph = use_cuda_graph and "cuda" in device
        self._tasks: dict[str, Callable[[], None]] = {}
        self._graph: wp.Graph | None = None
        self._flags: wp.array | None = None
        self._flags_host: np.ndarray | None = None
        self._state_dirty = True

        if model.shape_count > 0 and model.bvh_shapes is None:
            model.bvh_build_shapes(state)

    @property
    def model(self) -> newton.Model:
        """Newton model queried by managed sensors."""
        return self._model

    @property
    def state(self) -> newton.State:
        """Current Newton simulation state."""
        return self._state

    @property
    def task_names(self) -> tuple[str, ...]:
        """Names of registered sensor update tasks."""
        return tuple(self._tasks)

    @property
    def is_graph_captured(self) -> bool:
        """Whether sensor updates are currently backed by a captured graph."""
        return self._graph is not None

    def register(self, name: str, update_fn: Callable[[], None]) -> None:
        """Register a graph-capturable sensor update.

        Args:
            name: Unique task name.
            update_fn: Function that launches sensor work using stable buffers.

        Raises:
            ValueError: If ``name`` is already registered.
        """
        if name in self._tasks:
            raise ValueError(f"Newton sensor task '{name}' is already registered.")
        self._tasks[name] = update_fn
        self._invalidate_graph()

    def unregister(self, name: str) -> None:
        """Remove a sensor update task.

        Unknown task names are ignored so sensor cleanup remains idempotent.

        Args:
            name: Name of the task to remove.
        """
        if self._tasks.pop(name, None) is not None:
            self._invalidate_graph()

    def set_state(self, state: newton.State) -> None:
        """Bind the current simulation state and mark scene queries stale.

        Rebinding to a different state object invalidates the captured graph
        because captured Newton kernels retain the original array pointers.

        Args:
            state: Current Newton simulation state.
        """
        if state is not self._state:
            self._state = state
            self._invalidate_graph()
        self._state_dirty = True

    def update(self, *names: str) -> None:
        """Update the requested sensors.

        The scene BVH is refit first when :meth:`set_state` has marked the state
        dirty. With CUDA graphs enabled, refit and sensor tasks are conditionally
        selected inside the manager-owned graph.

        Args:
            names: Names of registered sensor tasks to update.

        Raises:
            KeyError: If a requested name is not registered.
        """
        for name in names:
            if name not in self._tasks:
                raise KeyError(f"Newton sensor task '{name}' is not registered.")

        if self._use_cuda_graph and self._graph is None:
            self._capture()
        if self._graph is None:
            if self._state_dirty:
                self._refit_bvh()
                self._state_dirty = False
            for name in names:
                self._tasks[name]()
            return

        assert self._flags_host is not None
        assert self._flags is not None
        self._flags_host.fill(0)
        self._flags_host[0] = int(self._state_dirty)
        task_names = tuple(self._tasks)
        for name in names:
            self._flags_host[1 + task_names.index(name)] = 1
        self._flags.assign(self._flags_host)
        wp.capture_launch(self._graph)
        self._state_dirty = False

    def clear(self) -> None:
        """Remove all sensor tasks and captured graph resources."""
        self._tasks.clear()
        self._invalidate_graph()

    def _refit_bvh(self) -> None:
        """Refit the model shape BVH against the current state."""
        if self._model.shape_count > 0 and self._model.bvh_shapes is not None:
            self._model.bvh_refit_shapes(self._state)

    def _capture(self) -> None:
        """Capture the BVH refit and registered sensor updates."""
        with wp.ScopedDevice(self._device):
            self._refit_bvh()
            for update_fn in self._tasks.values():
                update_fn()

        self._flags = wp.zeros(1 + len(self._tasks), dtype=wp.int32, device=self._device)
        self._flags_host = np.zeros(1 + len(self._tasks), dtype=np.int32)
        update_fns = tuple(self._tasks.values())

        def pipeline() -> None:
            assert self._flags is not None
            wp.capture_if(self._flags[0:1], self._refit_bvh)
            for index, update_fn in enumerate(update_fns):
                wp.capture_if(self._flags[index + 1 : index + 2], update_fn)

        self._graph = self._capture_fn(pipeline)
        if self._graph is None:
            self._use_cuda_graph = False
            logger.warning("Newton sensor graph capture failed; falling back to eager execution.")
        else:
            logger.info("Captured Newton sensor graph with %d task(s).", len(self._tasks))

    def _invalidate_graph(self) -> None:
        """Discard captured resources after task or state bindings change."""
        self._graph = None
        self._flags = None
        self._flags_host = None
