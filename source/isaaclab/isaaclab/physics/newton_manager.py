# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton physics manager for Isaac Lab."""

from __future__ import annotations

import numpy as np
import re
from typing import TYPE_CHECKING

import warp as wp
from newton import Axis, Contacts, Control, Model, ModelBuilder, State, eval_fk
from newton.examples import create_collision_pipeline
from newton.sensors import ContactSensor as NewtonContactSensor
from newton.sensors import populate_contacts
from newton.solvers import SolverBase, SolverFeatherstone, SolverMuJoCo, SolverNotifyFlags, SolverXPBD

from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.timer import Timer

from .physics_manager import PhysicsManager

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext
    from .newton_manager_cfg import NewtonManagerCfg
    from .solvers_cfg import NewtonSolverCfg


def flipped_match(x: str, y: str) -> re.Match | None:
    """Flipped match function.

    This function is used to match the contact partners' body/shape names with the body/shape names in the simulation.

    Args:
        x: The body/shape name in the simulation.
        y: The body/shape name in the contact view.

    Returns:
        The match object if the body/shape name is found in the contact view, otherwise None.
    """
    return re.match(y, x)


class NewtonManager(PhysicsManager):
    """Newton physics manager for Isaac Lab.

    This is a class-level (singleton-like) manager for the Newton simulation.
    It handles solver configuration, physics stepping, and reset.

    Lifecycle: initialize() -> reset() -> step() (repeated) -> close()
    """

    # Simulation context reference
    _sim: "SimulationContext | None" = None

    # Manager configuration
    _cfg: "NewtonManagerCfg | None" = None

    # Newton-specific state
    _builder: ModelBuilder = None
    _model: Model = None
    _device: str = "cuda:0"
    _dt: float = 1.0 / 200.0
    _solver_dt: float = 1.0 / 200.0
    _num_substeps: int = 1
    _solver = None
    _state_0: State = None
    _state_1: State = None
    _state_temp: State = None
    _control: Control = None
    _on_init_callbacks: list = []
    _on_start_callbacks: list = []
    _contacts: Contacts = None
    _needs_collision_pipeline: bool = False
    _collision_pipeline = None
    _newton_contact_sensor: NewtonContactSensor = None  # TODO: allow several contact sensors
    _report_contacts: bool = False
    _graph = None
    _newton_stage_path = None
    _sim_time = 0.0
    _usdrt_stage = None
    _newton_index_attr = "newton:index"
    _clone_physics_only = False
    _solver_type: str = "mujoco_warp"
    _gravity_vector: tuple[float, float, float] = (0.0, 0.0, -9.81)
    _up_axis: str = "Z"
    _num_envs: int | None = None
    _model_changes: set[int] = set()

    # ------------------------------------------------------------------
    # PhysicsManager Interface
    # ------------------------------------------------------------------

    @classmethod
    def initialize(cls, sim_context: "SimulationContext") -> None:
        """Initialize the manager with simulation context.

        Args:
            sim_context: Parent simulation context.
        """
        cls._sim = sim_context
        cls._cfg = sim_context.cfg.physics_manager_cfg  # type: ignore[assignment]

        # Set simulation parameters from config (device comes from sim_context, not cfg)
        cls._dt = cls._cfg.dt
        cls._device = sim_context.device
        cls._gravity_vector = cls._cfg.gravity

        # USD fabric sync only needed for OV rendering
        if cls._sim is not None:
            cls._clone_physics_only = "omniverse" not in cls._sim._visualizer_interface._visualizers_str  # type: ignore[union-attr]

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
        cls.forward_kinematics()

    @classmethod
    def step(cls) -> None:
        """Step the physics simulation."""
        if cls._sim is not None and cls._sim.is_playing():
            cls._step_internal()

    @classmethod
    def close(cls) -> None:
        """Clean up Newton physics resources."""
        cls.clear()
        cls._sim = None
        cls._cfg = None

    @classmethod
    def get_physics_dt(cls) -> float:
        """Get the physics timestep in seconds."""
        return cls._dt

    @classmethod
    def get_device(cls) -> str:
        """Get the physics simulation device."""
        return cls._device

    @classmethod
    def get_physics_sim_view(cls):
        """Get the physics simulation view (not applicable for Newton)."""
        return None

    @classmethod
    def is_fabric_enabled(cls) -> bool:
        """Check if fabric interface is enabled (not applicable for Newton)."""
        return False

    # ------------------------------------------------------------------
    # Newton-specific API
    # ------------------------------------------------------------------

    @classmethod
    def clear(cls):
        """Clear all state."""
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
        cls._sim_time = 0.0
        cls._on_init_callbacks = []
        cls._on_start_callbacks = []
        cls._usdrt_stage = None
        cls._up_axis = "Z"
        cls._model_changes = set()

    @classmethod
    def set_builder(cls, builder):
        cls._builder = builder

    @classmethod
    def add_on_init_callback(cls, callback) -> None:
        cls._on_init_callbacks.append(callback)

    @classmethod
    def add_on_start_callback(cls, callback) -> None:
        cls._on_start_callbacks.append(callback)

    @classmethod
    def add_model_change(cls, change: SolverNotifyFlags) -> None:
        cls._model_changes.add(change)

    @classmethod
    def start_simulation(cls) -> None:
        """Starts the simulation.

        This function finalizes the model and initializes the simulation state.
        """

        print(f"[INFO] Builder: {cls._builder}")
        if cls._builder is None:
            cls.instantiate_builder_from_stage()
        print("[INFO] Running on init callbacks")
        for callback in cls._on_init_callbacks:
            callback()
        print(f"[INFO] Finalizing model on device: {cls._device}")
        cls._builder.gravity = np.array(cls._gravity_vector)[-1]
        cls._builder.up_axis = Axis.from_string(cls._up_axis)
        with Timer(name="newton_finalize_builder", msg="Finalize builder took:", enable=True, format="ms"):
            cls._model = cls._builder.finalize(device=cls._device)
            cls._model.num_envs = cls._num_envs
        cls._state_0 = cls._model.state()
        cls._state_1 = cls._model.state()
        cls._state_temp = cls._model.state()
        cls._control = cls._model.control()
        cls.forward_kinematics()
        if cls._needs_collision_pipeline:
            cls._collision_pipeline = create_collision_pipeline(cls._model)
            cls._contacts = cls._model.collide(
                cls._state_0, collision_pipeline=cls._collision_pipeline
            )
        else:
            cls._contacts = Contacts(0, 0)
        print("[INFO] Running on start callbacks")
        for callback in cls._on_start_callbacks:
            callback()
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
    def instantiate_builder_from_stage(cls):
        from pxr import UsdGeom

        stage = get_current_stage()
        up_axis = UsdGeom.GetStageUpAxis(stage)
        builder = ModelBuilder(up_axis=up_axis)
        builder.add_usd(stage)
        cls.set_builder(builder)

    @classmethod
    def set_solver_settings(cls, cfg: "NewtonManagerCfg") -> None:
        """Set solver settings from config.

        Args:
            cfg: Newton manager configuration.
        """
        cls._cfg = cfg

    @classmethod
    def initialize_solver(cls):
        """Initializes the solver.

        This function initializes the solver based on the specified solver type. Currently, only XPBD and MuJoCoWarp
        are supported. The graphing of the simulation is performed in this function if the simulation is ran using
        a CUDA enabled device.

        .. warning::
            When using a CUDA enabled device, the simulation will be graphed. This means that this function steps the
            simulation once to capture the graph. Hence, this function should only be called after everything else in
            the simulation is initialized.
        """
        if cls._cfg is None:
            return

        with Timer(name="newton_initialize_solver", msg="Initialize solver took:", enable=True, format="ms"):
            cls._num_substeps = cls._cfg.num_substeps
            cls._solver_dt = cls._dt / cls._num_substeps
            print(cls._model.gravity)
            cls._solver = cls._get_solver(cls._model, cls._cfg.solver_cfg)
            if isinstance(cls._solver, SolverMuJoCo):
                use_mujoco_contacts = getattr(cls._cfg.solver_cfg, "use_mujoco_contacts", False)
                cls._needs_collision_pipeline = not use_mujoco_contacts
            else:
                cls._needs_collision_pipeline = True

        # Ensure we are using a CUDA enabled device
        assert cls._device.startswith("cuda"), "NewtonManager only supports CUDA enabled devices"

        # Capture the graph if CUDA is enabled
        with Timer(name="newton_cuda_graph", msg="CUDA graph took:", enable=True, format="ms"):
            if cls._cfg.use_cuda_graph:
                with wp.ScopedCapture() as capture:
                    cls.simulate()
                cls._graph = capture.graph

    @classmethod
    def simulate(cls) -> None:
        """Simulates the simulation.

        Performs one simulation step with the specified number of substeps. Depending on the solver type, this function
        may need to explicitly compute the collisions. This function also aggregates the contacts and evaluates the
        contact sensors. Finally, it performs the state swapping for Newton.
        """
        state_0_dict = cls._state_0.__dict__
        state_1_dict = cls._state_1.__dict__
        state_temp_dict = cls._state_temp.__dict__
        contacts = None

        # MJWarp computes its own collisions.
        if cls._needs_collision_pipeline:
            contacts = cls._model.collide(
                cls._state_0, collision_pipeline=cls._collision_pipeline
            )

        if cls._num_substeps % 2 == 0:
            for i in range(cls._num_substeps):
                cls._solver.step(
                    cls._state_0,
                    cls._state_1,
                    cls._control,
                    contacts,
                    cls._solver_dt,
                )
                cls._state_0, cls._state_1 = cls._state_1, cls._state_0
                cls._state_0.clear_forces()
        else:
            for i in range(cls._num_substeps):
                cls._solver.step(
                    cls._state_0,
                    cls._state_1,
                    cls._control,
                    contacts,
                    cls._solver_dt,
                )

                # FIXME: Ask Lukasz help to deal with non-even number of substeps. This should not be needed.
                if i < cls._num_substeps - 1 or not cls._cfg.use_cuda_graph:
                    # we can just swap the state references
                    cls._state_0, cls._state_1 = cls._state_1, cls._state_0
                elif cls._cfg.use_cuda_graph:
                    # swap states by actually copying the state arrays to make sure the graph capture works
                    for key, value in state_0_dict.items():
                        if isinstance(value, wp.array):  # type: ignore[arg-type]
                            if key not in state_temp_dict:
                                state_temp_dict[key] = wp.empty_like(value)
                            state_temp_dict[key].assign(value)
                            state_0_dict[key].assign(state_1_dict[key])
                            state_1_dict[key].assign(state_temp_dict[key])
                cls._state_0.clear_forces()

        if cls._report_contacts:
            populate_contacts(cls._contacts, cls._solver)
            cls._newton_contact_sensor.eval(cls._contacts)

    @classmethod
    def set_device(cls, device: str) -> None:
        """Sets the device to use for the Newton simulation.

        Args:
            device (str): The device to use for the Newton simulation.
        """
        cls._device = device

    @classmethod
    def _step_internal(cls) -> None:
        """Steps the simulation (internal).

        This function steps the simulation by the specified time step in the simulation configuration.
        """
        if cls._model_changes:
            for change in cls._model_changes:
                cls._solver.notify_model_changed(change)
            cls._model_changes = set()

        if cls._cfg is not None and cls._cfg.use_cuda_graph:
            wp.capture_launch(cls._graph)  # type: ignore[arg-type]
        else:
            cls.simulate()

        if cls._cfg is not None and cls._cfg.debug_mode:
            convergence_data = cls.get_solver_convergence_steps()
            # print(f"solver niter: {convergence_data}")
            if convergence_data["max"] == cls._solver.mjw_model.opt.iterations:
                print("solver didn't converge!", convergence_data["max"])

        cls._sim_time += cls._solver_dt * cls._num_substeps

    @classmethod
    def get_solver_convergence_steps(cls) -> dict[str, float | int]:
        niter = cls._solver.mjw_data.solver_niter.numpy()
        max_niter = np.max(niter)
        mean_niter = np.mean(niter)
        min_niter = np.min(niter)
        std_niter = np.std(niter)
        return {"max": max_niter, "mean": mean_niter, "min": min_niter, "std": std_niter}

    @classmethod
    def set_simulation_dt(cls, dt: float) -> None:
        """Sets the simulation time step and the number of substeps.

        Args:
            dt (float): The simulation time step.
        """
        cls._dt = dt

    @classmethod
    def get_model(cls):
        return cls._model

    @classmethod
    def get_state_0(cls):
        return cls._state_0

    @classmethod
    def get_state_1(cls):
        return cls._state_1

    @classmethod
    def get_control(cls):
        return cls._control

    @classmethod
    def get_dt(cls):
        return cls._dt

    @classmethod
    def get_solver_dt(cls):
        return cls._solver_dt

    @classmethod
    def forward_kinematics(cls, mask: wp.array | None = None) -> None:
        """Evaluates the forward kinematics for the selected articulations.

        This function evaluates the forward kinematics for the selected articulations.
        """
        eval_fk(
            cls._model,
            cls._state_0.joint_q,
            cls._state_0.joint_qd,
            cls._state_0,
            None,
        )

    @classmethod
    def _get_solver(cls, model: Model, solver_cfg: "NewtonSolverCfg") -> SolverBase:
        """Create and return the appropriate solver based on config.

        Args:
            model: The Newton model.
            solver_cfg: Solver configuration.

        Returns:
            The initialized solver.
        """
        # Convert config to dict if needed (configclass adds to_dict method)
        if hasattr(solver_cfg, "to_dict"):
            cfg_dict = solver_cfg.to_dict()  # type: ignore[union-attr]
        else:
            cfg_dict = dict(solver_cfg) if isinstance(solver_cfg, dict) else {}  # type: ignore[arg-type]

        cls._solver_type = cfg_dict.pop("solver_type", "mujoco_warp")

        if cls._solver_type == "mujoco_warp":
            return SolverMuJoCo(model, **cfg_dict)
        elif cls._solver_type == "xpbd":
            return SolverXPBD(model, **cfg_dict)
        elif cls._solver_type == "featherstone":
            return SolverFeatherstone(model, **cfg_dict)
        else:
            raise ValueError(f"Invalid solver type: {cls._solver_type}")

    @classmethod
    def add_contact_sensor(
        cls,
        body_names_expr: str | list[str] | None = None,
        shape_names_expr: str | list[str] | None = None,
        contact_partners_body_expr: str | list[str] | None = None,
        contact_partners_shape_expr: str | list[str] | None = None,
        prune_noncolliding: bool = False,
        verbose: bool = False,
    ):
        """Adds a contact view.

        Adds a contact view to the simulation allowing to report contacts between the specified bodies/shapes and the
        contact partners. As of now, only one body/shape name expression can be provided. Similarly, only one contact
        partner body/shape expression can be provided. If no contact partner expression is provided, the contact view
        will report contacts with all bodies/shapes.

        Note that we make an explicit difference between a body and a shape. A body is a rigid body, while a shape
        is a collision shape. A body can have multiple shapes. The shape option allows a more fine-grained control
        over the contact reporting.

        Args:
            body_names_expr (str | None): The expression for the body names.
            shape_names_expr (str | None): The expression for the shape names.
            contact_partners_body_expr (str | None): The expression for the contact partners' body names.
            contact_partners_shape_expr (str | None): The expression for the contact partners' shape names.
            prune_noncolliding (bool): Make the force matrix sparse using the collision pairs in the model.
            verbose (bool): Whether to print verbose information.
        """
        if body_names_expr is None and shape_names_expr is None:
            raise ValueError("At least one of body_names_expr or shape_names_expr must be provided")
        if body_names_expr is not None and shape_names_expr is not None:
            raise ValueError("Only one of body_names_expr or shape_names_expr must be provided")
        if contact_partners_body_expr is not None and contact_partners_shape_expr is not None:
            raise ValueError("Only one of contact_partners_body_expr or contact_partners_shape_expr must be provided")
        if contact_partners_body_expr is None and contact_partners_shape_expr is None:
            print(f"[INFO] Adding contact view for {body_names_expr}. It will report contacts with all bodies/shapes.")
        else:
            if body_names_expr is not None:
                if contact_partners_body_expr is not None:
                    print(f"[INFO] Adding contact view for {body_names_expr} with filter {contact_partners_body_expr}.")
                else:
                    print(f"[INFO] Adding contact view for {body_names_expr} with filter {shape_names_expr}.")
            else:
                if contact_partners_body_expr is not None:
                    print(
                        f"[INFO] Adding contact view for {shape_names_expr} with filter {contact_partners_body_expr}."
                    )
                else:
                    print(
                        f"[INFO] Adding contact view for {shape_names_expr} with filter {contact_partners_shape_expr}."
                    )

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
