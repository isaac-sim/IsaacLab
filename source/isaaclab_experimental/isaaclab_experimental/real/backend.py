# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal lifecycle adapter for the experimental real robot backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from isaaclab.physics import PhysicsEvent, PhysicsManager
from isaaclab.scene_data import SceneDataBackend, SceneDataFormat

if TYPE_CHECKING:
    from isaaclab.sim import SimulationContext

    from .backend_cfg import RealCfg
    from .session import DeploymentSession


class _EmptySceneDataBackend(SceneDataBackend):
    """Publish an empty scene because this prototype has no rendering consumers."""

    transforms_timestamp = 0

    def __init__(self) -> None:
        self._transforms = SceneDataFormat.Transform()
        self._transforms.transforms = wp.zeros(0, dtype=wp.transformf, device="cpu")

    @property
    def transforms(self) -> SceneDataFormat.Transform:
        return self._transforms

    @property
    def transform_count(self) -> int:
        return 0

    @property
    def transform_paths(self) -> list[str]:
        return []


class RealManager(PhysicsManager):
    """Own one deployment session without advancing a dynamics solver."""

    backend_name = "real"
    _cfg: RealCfg

    def __init__(self, cfg: RealCfg) -> None:
        super().__init__(cfg)
        self._session: DeploymentSession | None = None

    @property
    def session(self) -> DeploymentSession:
        """The connected session shared by the robot and the backend lifecycle."""
        if self._session is None:
            raise RuntimeError("The real backend deployment session is not initialized.")
        return self._session

    def initialize(self, sim_context: SimulationContext) -> None:
        super().initialize(sim_context)
        if sim_context.cfg.device != "cpu":
            raise ValueError("The real backend prototype requires device='cpu'.")
        if sim_context.cfg.visualizer_cfgs:
            raise ValueError("The real backend prototype requires visualizer_cfgs=[].")

        from isaaclab.assets import Articulation  # noqa: PLC0415

        from .articulation import RealArticulation  # noqa: PLC0415

        Articulation.register("real", RealArticulation)
        self._session = self._cfg.session_factory()

    def reset(self, soft: bool = False) -> None:
        """Make telemetry available to assets without resetting the physical robot."""
        self.forward()
        self.dispatch_event(PhysicsEvent.PHYSICS_READY)

    def forward(self) -> None:
        """Keep the session passive; articulation updates read its latest telemetry."""

    def step(self) -> None:
        """Complete one session tick; the driver owns any faster servo loop."""
        dt = self.get_physics_dt()
        self.session.step(dt)
        self._sim_time += dt

    def get_scene_data_backend(self) -> SceneDataBackend:
        return _EmptySceneDataBackend()

    def close(self) -> None:
        """Invalidate asset handles before closing their shared session."""
        try:
            super().close()
        finally:
            session, self._session = self._session, None
            if session is not None:
                session.close()
