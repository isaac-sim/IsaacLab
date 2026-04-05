# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone Newton physics manager — no IsaacLab dependencies."""

import inspect

import warp as wp

import newton
from newton import CollisionPipeline
from newton.solvers import SolverMuJoCo, SolverNotifyFlags


def _to_kwargs(cfg, target_cls):
    """Extract valid __init__ kwargs from a namespace/dict config."""
    d = dict(vars(cfg)) if hasattr(cfg, "__dict__") else dict(cfg)
    valid = set(inspect.signature(target_cls.__init__).parameters.keys()) - {"self", "model"}
    return {k: v for k, v in d.items() if k in valid}


class NewtonSim:
    """Minimal Newton simulation: model + solver + collision + viewer sync."""

    def __init__(self, builder, solver_cfg, collision_cfg, physics_dt, num_substeps=1, device="cuda:0"):
        """Initialize Newton simulation from a finalized builder.

        Args:
            builder: Newton ModelBuilder (already populated with worlds).
            solver_cfg: Namespace/dict with SolverMuJoCo kwargs (njmax, nconmax, cone, etc.).
            collision_cfg: Namespace/dict with CollisionPipeline kwargs, or None for MuJoCo contacts.
            physics_dt: Physics timestep [s].
            num_substeps: Number of solver substeps per step.
            device: Torch device string.
        """
        self.device = device
        self._graph = None
        self._num_substeps = num_substeps
        self._solver_dt = physics_dt / num_substeps

        # Finalize model
        self.model = builder.finalize(device=device)
        self.state = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

        # Solver
        solver_kwargs = _to_kwargs(solver_cfg, SolverMuJoCo)
        solver_kwargs.pop("solver_type", None)
        self.solver = SolverMuJoCo(self.model, **solver_kwargs)

        # Collision pipeline (None = MuJoCo handles contacts internally)
        use_mujoco_contacts = getattr(solver_cfg, "use_mujoco_contacts", True)
        self.collision_pipeline = None
        self.contacts = None

        if not use_mujoco_contacts and collision_cfg is not None:
            col_kwargs = _to_kwargs(collision_cfg, CollisionPipeline)
            col_kwargs.pop("sdf_hydroelastic_config", None)
            self.collision_pipeline = CollisionPipeline(self.model, **col_kwargs)
            self.contacts = self.collision_pipeline.contacts()

    def _simulate(self):
        """One physics step: collide → solve × substeps → clear forces."""
        if self.collision_pipeline is not None:
            self.collision_pipeline.collide(self.state, self.contacts)
        for _ in range(self._num_substeps):
            self.solver.step(self.state, self.state, self.control, self.contacts, self._solver_dt)
            self.state.clear_forces()

    def step(self):
        """Step physics. Uses CUDA graph if captured, otherwise eager."""
        with wp.ScopedDevice(self.device):
            if self._graph is not None:
                wp.capture_launch(self._graph)
            else:
                self._simulate()

    def forward(self):
        """Sync body transforms from joint state (call before rendering)."""
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

    def notify_model_changed(self, flags=None):
        """Sync Newton model changes to the MuJoCo Warp solver.

        Must be called after modifying model arrays (joint_armature,
        joint_target_ke, joint_target_kd, joint_friction, joint_effort_limit,
        etc.) so the solver picks up the new values.

        Args:
            flags: SolverNotifyFlags bitmask. Defaults to JOINT_DOF_PROPERTIES.
        """
        if flags is None:
            flags = SolverNotifyFlags.JOINT_DOF_PROPERTIES
        self.solver.notify_model_changed(flags)

    def capture_graph(self):
        """Capture CUDA graph for faster stepping. Falls back to eager on failure."""
        with wp.ScopedDevice(self.device):
            try:
                with wp.ScopedCapture() as capture:
                    self._simulate()
                self._graph = capture.graph
            except Exception:
                self._graph = None
