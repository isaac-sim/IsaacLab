# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton physics manager for Isaac Lab."""

from __future__ import annotations

import logging
import numpy as np
import re
from typing import TYPE_CHECKING

import warp as wp
from newton import Axis, Contacts, Control, Model, ModelBuilder, State, eval_fk
from newton.examples import create_collision_pipeline
from newton.sensors import SensorContact as NewtonContactSensor
from newton.sensors import populate_contacts
from newton.solvers import SolverBase, SolverFeatherstone, SolverMuJoCo, SolverNotifyFlags, SolverXPBD

from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.timer import Timer

from .physics_manager import PhysicsManager, PhysicsEvent

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext
    from .newton_manager_cfg import NewtonManagerCfg

logger = logging.getLogger(__name__)


def flipped_match(x: str, y: str) -> bool:
    """Flipped match function for contact partner matching.

    Args:
        x: The body/shape name in the simulation.
        y: The body/shape name in the contact view.

    Returns:
        True if the body/shape name matches the contact view pattern.
    """
    return re.match(y, x) is not None


class NewtonManager(PhysicsManager):
    """Newton physics manager for Isaac Lab.

    This is a class-level (singleton-like) manager for the Newton simulation.
    It handles solver configuration, physics stepping, and reset.

    Lifecycle: initialize() -> reset() -> step() (repeated) -> close()
    """

    _cfg: "NewtonManagerCfg | None" = None

    _solver_dt: float = 1.0 / 200.0
    _num_substeps: int = 1
    _num_envs: int | None = None

    # Newton model and state
    _builder: ModelBuilder = None
    _model: Model = None
    _solver: SolverBase = None
    _solver_type: str = "mujoco_warp"
    _state_0: State = None
    _state_1: State = None
    _state_temp: State = None
    _control: Control = None

    # Physics settings
    _gravity_vector: tuple[float, float, float] = (0.0, 0.0, -9.81)
    _up_axis: str = "Z"

    # Collision and contacts
    _contacts: Contacts = None
    _needs_collision_pipeline: bool = False
    _collision_pipeline = None
    _newton_contact_sensor: NewtonContactSensor = None
    _report_contacts: bool = False

    # CUDA graphing
    _graph = None

    # USD/Fabric sync
    _newton_stage_path = None
    _usdrt_stage = None
    _newton_index_attr = "newton:index"
    _clone_physics_only = False

    # Model changes (callbacks use unified system from PhysicsManager)
    _model_changes: set[int] = set()

    @classmethod
    def initialize(cls, sim_context: "SimulationContext") -> None:
        """Initialize the manager with simulation context.

        Args:
            sim_context: Parent simulation context.
        """
        super().initialize(sim_context)

        # Newton-specific setup
        cls._device = cls._cfg.device  # type: ignore[union-attr]
        cls._gravity_vector = cls._cfg.gravity  # type: ignore[union-attr]

        # USD fabric sync only needed for OV rendering
        viz_str = cls._sim.get_setting("/isaaclab/visualizer") or ""  # type: ignore[union-attr]
        requested = [v.strip() for v in viz_str.split(",") if v.strip()]
        cls._clone_physics_only = "omniverse" not in requested

    @classmethod
    def reset(cls, soft: bool = False) -> None:
        """Reset physics simulation.

        Args:
            soft: If True, skip full reinitialization.
        """
        if not soft:
            cls.start_simulation()
            cls.initialize_solver()

    @classmethod
    def forward(cls) -> None:
        """Update articulation kinematics without stepping physics."""
        eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, None)

    @classmethod
    def step(cls) -> None:
        """Step the physics simulation."""
        if cls._sim is None or not cls._sim.is_playing():
            return

        # Notify solver of model changes
        if cls._model_changes:
            for change in cls._model_changes:
                cls._solver.notify_model_changed(change)
            cls._model_changes = set()

        # Step simulation (graphed or not)
        if cls._cfg is not None and cls._cfg.use_cuda_graph:
            wp.capture_launch(cls._graph)  # type: ignore[arg-type]
        else:
            cls._simulate()

        # Debug convergence info
        if cls._cfg is not None and cls._cfg.debug_mode:
            convergence_data = cls.get_solver_convergence_steps()
            if convergence_data["max"] == cls._solver.mjw_model.opt.iterations:
                logger.warning(f"Solver didn't converge! max_iter={convergence_data['max']}")

        cls._sim_time += cls._solver_dt * cls._num_substeps

    @classmethod
    def close(cls) -> None:
        """Clean up Newton physics resources."""
        cls.clear()
        super().close()

    # get_physics_dt(), get_device(), get_physics_sim_view(), is_fabric_enabled()
    # are inherited from PhysicsManager base class

    @classmethod
    def clear(cls):
        """Clear all Newton-specific state (callbacks cleared by super().close())."""
        cls._builder = None
        cls._model = None
        cls._solver = None
        cls._state_0 = None
        cls._state_1 = None
        cls._state_temp = None
        cls._control = None
        cls._contacts = None
        cls._needs_collision_pipeline = False
        cls._collision_pipeline = None
        cls._newton_contact_sensor = None
        cls._report_contacts = False
        cls._graph = None
        cls._newton_stage_path = None
        cls._usdrt_stage = None
        cls._up_axis = "Z"
        cls._model_changes = set()

    @classmethod
    def set_builder(cls, builder: ModelBuilder) -> None:
        """Set the Newton model builder."""
        cls._builder = builder

    @classmethod
    def add_model_change(cls, change: SolverNotifyFlags) -> None:
        """Register a model change to notify the solver."""
        cls._model_changes.add(change)

    @classmethod
    def start_simulation(cls) -> None:
        """Start simulation by finalizing model and initializing state."""
        logger.debug(f"Builder: {cls._builder}")

        # Create builder from USD stage if not provided
        if cls._builder is None:
            from pxr import UsdGeom
            stage = get_current_stage()
            up_axis = UsdGeom.GetStageUpAxis(stage)
            builder = ModelBuilder(up_axis=up_axis)
            builder.add_usd(stage)
            cls._builder = builder

        logger.info("Dispatching MODEL_INIT callbacks")
        cls.dispatch_event(PhysicsEvent.MODEL_INIT)

        logger.info(f"Finalizing model on device: {cls._device}")
        cls._builder.gravity = np.array(cls._gravity_vector)[-1]
        cls._builder.up_axis = Axis.from_string(cls._up_axis)

        with Timer(name="newton_finalize_builder", msg="Finalize builder took:", enable=True, format="ms"):
            cls._model = cls._builder.finalize(device=cls._device)
            cls._model.num_envs = cls._num_envs

        cls._state_0 = cls._model.state()
        cls._state_1 = cls._model.state()
        cls._state_temp = cls._model.state()
        cls._control = cls._model.control()
        eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, None)

        # Initialize collision pipeline if needed
        if cls._needs_collision_pipeline:
            cls._collision_pipeline = create_collision_pipeline(cls._model)
            cls._contacts = cls._model.collide(cls._state_0, collision_pipeline=cls._collision_pipeline)
        else:
            cls._contacts = Contacts(0, 0)

        logger.info("Dispatching SIMULATION_START callbacks")
        cls.dispatch_event(PhysicsEvent.PHYSICS_READY)

        # Setup USD/Fabric sync for Omniverse rendering
        if not cls._clone_physics_only:
            import usdrt
            cls._usdrt_stage = get_current_stage(fabric=True)
            for i, prim_path in enumerate(cls._model.body_key):
                prim = cls._usdrt_stage.GetPrimAtPath(prim_path)
                prim.CreateAttribute(cls._newton_index_attr, usdrt.Sdf.ValueTypeNames.UInt, True)
                prim.GetAttribute(cls._newton_index_attr).Set(i)
                xformable_prim = usdrt.Rt.Xformable(prim)
                if not xformable_prim.HasWorldXform():
                    xformable_prim.SetWorldXformFromUsd()

    @classmethod
    def initialize_solver(cls) -> None:
        """Initialize the solver and optionally capture CUDA graph."""
        if cls._cfg is None:
            return

        with Timer(name="newton_initialize_solver", msg="Initialize solver took:", enable=True, format="ms"):
            cls._num_substeps = cls._cfg.num_substeps
            cls._solver_dt = cls.get_physics_dt() / cls._num_substeps

            # Create solver from config
            solver_cfg = cls._cfg.solver_cfg
            cfg_dict = solver_cfg.to_dict() if hasattr(solver_cfg, "to_dict") else {}
            cls._solver_type = cfg_dict.pop("solver_type", "mujoco_warp")

            if cls._solver_type == "mujoco_warp":
                cls._solver = SolverMuJoCo(cls._model, **cfg_dict)
            elif cls._solver_type == "xpbd":
                cls._solver = SolverXPBD(cls._model, **cfg_dict)
            elif cls._solver_type == "featherstone":
                cls._solver = SolverFeatherstone(cls._model, **cfg_dict)
            else:
                raise ValueError(f"Invalid solver type: {cls._solver_type}")

            # Determine if we need collision pipeline
            if isinstance(cls._solver, SolverMuJoCo):
                use_mujoco_contacts = getattr(cls._cfg.solver_cfg, "use_mujoco_contacts", False)
                cls._needs_collision_pipeline = not use_mujoco_contacts
            else:
                cls._needs_collision_pipeline = True

        # Capture CUDA graph for performance
        assert cls._device.startswith("cuda"), "NewtonManager only supports CUDA enabled devices"

        with Timer(name="newton_cuda_graph", msg="CUDA graph took:", enable=True, format="ms"):
            if cls._cfg.use_cuda_graph:
                with wp.ScopedCapture() as capture:
                    cls._simulate()
                cls._graph = capture.graph

    @classmethod
    def _simulate(cls) -> None:
        """Run one simulation step with substeps."""
        state_0_dict = cls._state_0.__dict__
        state_1_dict = cls._state_1.__dict__
        state_temp_dict = cls._state_temp.__dict__
        contacts = None

        # MJWarp computes its own collisions
        if cls._needs_collision_pipeline:
            contacts = cls._model.collide(cls._state_0, collision_pipeline=cls._collision_pipeline)

        if cls._num_substeps % 2 == 0:
            for _ in range(cls._num_substeps):
                cls._solver.step(cls._state_0, cls._state_1, cls._control, contacts, cls._solver_dt)
                cls._state_0, cls._state_1 = cls._state_1, cls._state_0
                cls._state_0.clear_forces()
        else:
            for i in range(cls._num_substeps):
                cls._solver.step(cls._state_0, cls._state_1, cls._control, contacts, cls._solver_dt)

                if i < cls._num_substeps - 1 or not cls._cfg.use_cuda_graph:
                    cls._state_0, cls._state_1 = cls._state_1, cls._state_0
                elif cls._cfg.use_cuda_graph:
                    # Swap states by copying arrays for CUDA graph compatibility
                    for key, value in state_0_dict.items():
                        if isinstance(value, wp.array):  # type: ignore[arg-type]
                            if key not in state_temp_dict:
                                state_temp_dict[key] = wp.empty_like(value)
                            state_temp_dict[key].assign(value)
                            state_0_dict[key].assign(state_1_dict[key])
                            state_1_dict[key].assign(state_temp_dict[key])
                cls._state_0.clear_forces()

        # Populate contacts for contact sensors
        if cls._report_contacts:
            populate_contacts(cls._contacts, cls._solver)
            cls._newton_contact_sensor.eval(cls._contacts)

    @classmethod
    def get_solver_convergence_steps(cls) -> dict[str, float | int]:
        """Get solver convergence statistics."""
        niter = cls._solver.mjw_data.solver_niter.numpy()
        return {
            "max": np.max(niter),
            "mean": np.mean(niter),
            "min": np.min(niter),
            "std": np.std(niter),
        }

    # State accessors (used extensively by articulation/rigid object data)
    @classmethod
    def get_model(cls) -> Model:
        """Get the Newton model."""
        return cls._model

    @classmethod
    def get_state_0(cls) -> State:
        """Get the current state."""
        return cls._state_0

    @classmethod
    def get_state_1(cls) -> State:
        """Get the next state."""
        return cls._state_1

    @classmethod
    def get_control(cls) -> Control:
        """Get the control object."""
        return cls._control

    @classmethod
    def get_dt(cls) -> float:
        """Get the physics timestep. Alias for get_physics_dt()."""
        return cls.get_physics_dt()

    @classmethod
    def get_solver_dt(cls) -> float:
        """Get the solver substep timestep."""
        return cls._solver_dt

    @classmethod
    def add_contact_sensor(
        cls,
        body_names_expr: str | list[str] | None = None,
        shape_names_expr: str | list[str] | None = None,
        contact_partners_body_expr: str | list[str] | None = None,
        contact_partners_shape_expr: str | list[str] | None = None,
        prune_noncolliding: bool = False,
        verbose: bool = False,
    ) -> None:
        """Add a contact sensor for reporting contacts between bodies/shapes.

        Note: Only one contact sensor can be active at a time.

        Args:
            body_names_expr: Expression for body names to sense.
            shape_names_expr: Expression for shape names to sense.
            contact_partners_body_expr: Expression for contact partner body names.
            contact_partners_shape_expr: Expression for contact partner shape names.
            prune_noncolliding: Make force matrix sparse using collision pairs.
            verbose: Print verbose information.
        """
        # Validate inputs
        if body_names_expr is None and shape_names_expr is None:
            raise ValueError("At least one of body_names_expr or shape_names_expr must be provided")
        if body_names_expr is not None and shape_names_expr is not None:
            raise ValueError("Only one of body_names_expr or shape_names_expr must be provided")
        if contact_partners_body_expr is not None and contact_partners_shape_expr is not None:
            raise ValueError("Only one of contact_partners_body_expr or contact_partners_shape_expr must be provided")

        # Log sensor configuration
        sensor_target = body_names_expr or shape_names_expr
        partner_filter = contact_partners_body_expr or contact_partners_shape_expr or "all bodies/shapes"
        logger.info(f"Adding contact sensor for {sensor_target} with filter {partner_filter}")

        cls._newton_contact_sensor = NewtonContactSensor(
            cls._model,
            sensing_obj_bodies=body_names_expr,
            sensing_obj_shapes=shape_names_expr,
            counterpart_bodies=contact_partners_body_expr,
            counterpart_shapes=contact_partners_shape_expr,
            match_fn=flipped_match,
            include_total=True,
            prune_noncolliding=prune_noncolliding,
            verbose=verbose,
        )
        cls._report_contacts = True
