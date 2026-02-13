# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import numpy as np
import re

import warp as wp
from newton import Axis, BroadPhaseMode, CollisionPipeline, Contacts, Control, Model, ModelBuilder, State, eval_fk
from newton.sensors import SensorContact as NewtonContactSensor
from newton.solvers import SolverBase, SolverFeatherstone, SolverMuJoCo, SolverNotifyFlags, SolverXPBD

from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.timer import Timer

logger = logging.getLogger(__name__)


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


class NewtonManager:
    _builder: ModelBuilder = None
    _model: Model = None
    _device: str = "cuda:0"
    _dt: float = 1.0 / 200.0
    _solver_dt: float = 1.0 / 200.0
    _num_substeps: int = 1
    _use_single_state: bool | None = None
    """Use only one state for both input and output for solver stepping. Requires solver support."""
    _solver = None
    _state_0: State = None
    _state_1: State = None
    _control: Control = None
    _on_init_callbacks: list = []
    _on_start_callbacks: list = []
    _contacts: Contacts | None = None
    _needs_collision_pipeline: bool = False
    _collision_pipeline = None
    _newton_contact_sensors: dict = {}  # Maps sensor_key to NewtonContactSensor
    _report_contacts: bool = False
    _graph = None
    _newton_stage_path = None
    _sim_time = 0.0
    _usdrt_stage = None
    _newton_index_attr = "newton:index"
    _clone_physics_only = False
    _cfg: NewtonCfg = NewtonCfg()
    _solver_type: str = "mujoco_warp"
    _gravity_vector: tuple[float, float, float] = (0.0, 0.0, -9.81)
    _up_axis: str = "Z"
    _num_envs: int = None
    _model_changes: set[int] = set()

    @classmethod
    def clear(cls):
        NewtonManager._builder = None
        NewtonManager._model = None
        NewtonManager._solver = None
        NewtonManager._use_single_state = None
        NewtonManager._state_0 = None
        NewtonManager._state_1 = None
        NewtonManager._control = None
        NewtonManager._contacts = None
        NewtonManager._needs_collision_pipeline = False
        NewtonManager._collision_pipeline = None
        NewtonManager._newton_contact_sensors = {}
        NewtonManager._report_contacts = False
        NewtonManager._graph = None
        NewtonManager._newton_stage_path = None
        NewtonManager._sim_time = 0.0
        NewtonManager._on_init_callbacks = []
        NewtonManager._on_start_callbacks = []
        NewtonManager._usdrt_stage = None
        # Only create new config if not during Python shutdown
        try:
            NewtonManager._cfg = NewtonCfg()
        except (ImportError, AttributeError, TypeError):
            NewtonManager._cfg = None
        NewtonManager._up_axis = "Z"
        NewtonManager._first_call = True
        NewtonManager._model_changes = set()

    @classmethod
    def set_builder(cls, builder):
        NewtonManager._builder = builder

    @classmethod
    def add_on_init_callback(cls, callback) -> None:
        NewtonManager._on_init_callbacks.append(callback)

    @classmethod
    def add_on_start_callback(cls, callback) -> None:
        NewtonManager._on_start_callbacks.append(callback)

    @classmethod
    def add_model_change(cls, change: SolverNotifyFlags) -> None:
        NewtonManager._model_changes.add(change)

    @classmethod
    def start_simulation(cls) -> None:
        """Starts the simulation.

        This function finalizes the model and initializes the simulation state.
        Note: Collision pipeline is initialized later in initialize_solver() after
        we determine whether the solver needs external collision detection.
        """

        print(f"[INFO] Builder: {NewtonManager._builder}")
        if NewtonManager._builder is None:
            NewtonManager.instantiate_builder_from_stage()
        print("[INFO] Running on init callbacks")
        for callback in NewtonManager._on_init_callbacks:
            callback()
        print(f"[INFO] Finalizing model on device: {NewtonManager._device}")
        NewtonManager._builder.up_axis = Axis.from_string(NewtonManager._up_axis)
        # Set smaller contact margin for manipulation examples (default 10cm is too large)
        NewtonManager._builder.default_shape_cfg.contact_margin = 0.01
        with Timer(name="newton_finalize_builder", msg="Finalize builder took:", enable=True, format="ms"):
            NewtonManager._model = NewtonManager._builder.finalize(device=NewtonManager._device)
            NewtonManager._model.set_gravity(NewtonManager._gravity_vector)
            NewtonManager._model.num_envs = NewtonManager._num_envs

        NewtonManager._state_0 = NewtonManager._model.state()
        NewtonManager._state_1 = NewtonManager._model.state()
        NewtonManager._control = NewtonManager._model.control()
        NewtonManager.forward_kinematics()

        print("[INFO] Running on start callbacks")
        for callback in NewtonManager._on_start_callbacks:
            callback()
        if not NewtonManager._clone_physics_only:
            import usdrt

            NewtonManager._usdrt_stage = get_current_stage(fabric=True)
            for i, prim_path in enumerate(NewtonManager._model.body_key):
                prim = NewtonManager._usdrt_stage.GetPrimAtPath(prim_path)
                prim.CreateAttribute(NewtonManager._newton_index_attr, usdrt.Sdf.ValueTypeNames.UInt, True)
                prim.GetAttribute(NewtonManager._newton_index_attr).Set(i)
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
        NewtonManager.set_builder(builder)

    @classmethod
    def set_solver_settings(cls, newton_params: dict):
        NewtonManager._cfg = NewtonCfg(**newton_params)

    @classmethod
    def _initialize_contacts(cls) -> None:
        """Unified method to initialize contacts and collision pipeline.

        This method handles both Newton collision pipeline and MuJoCo contact modes.
        It ensures contacts are properly initialized with force attributes if sensors are registered.
        """
        if NewtonManager._needs_collision_pipeline:
            # Newton collision pipeline: create pipeline and generate contacts
            if NewtonManager._collision_pipeline is None:
                NewtonManager._collision_pipeline = CollisionPipeline(
                    NewtonManager._model, broad_phase_mode=BroadPhaseMode.EXPLICIT
                )
            if NewtonManager._contacts is None:
                NewtonManager._contacts = NewtonManager._collision_pipeline.contacts()

        elif NewtonManager._solver is not None and isinstance(NewtonManager._solver, SolverMuJoCo):
            # MuJoCo contacts mode: create properly sized Contacts object
            # The solver's update_contacts() will populate this from MuJoCo data
            rigid_contact_max = NewtonManager._solver.get_max_contact_count()
            NewtonManager._contacts = Contacts(
                rigid_contact_max=rigid_contact_max,
                soft_contact_max=0,
                device=NewtonManager._device,
                requested_attributes=NewtonManager._model.get_requested_contact_attributes(),
            )

    @classmethod
    def initialize_solver(cls):
        """Initializes the solver and collision pipeline.

        This function initializes the solver based on the specified solver type. Currently, only XPBD and MuJoCoWarp
        are supported. If the solver requires external collision detection (i.e., not using MuJoCo's internal
        contacts), a unified collision pipeline is created.

        The graphing of the simulation is performed in this function if the simulation is ran using
        a CUDA enabled device.

        .. warning::
            When using a CUDA enabled device, the simulation will be graphed. This means that this function steps the
            simulation once to capture the graph. Hence, this function should only be called after everything else in
            the simulation is initialized.
        """
        with Timer(name="newton_initialize_solver", msg="Initialize solver took:", enable=True, format="ms"):
            NewtonManager._num_substeps = NewtonManager._cfg.num_substeps
            NewtonManager._solver_dt = NewtonManager._dt / NewtonManager._num_substeps
            NewtonManager._solver = NewtonManager._get_solver(NewtonManager._model, NewtonManager._cfg.solver_cfg)

            # Determine if we need external collision detection
            # - SolverMuJoCo with use_mujoco_contacts=True: uses internal MuJoCo collision detection
            # - SolverMuJoCo with use_mujoco_contacts=False: needs Newton's unified collision pipeline
            # - Other solvers (XPBD, Featherstone): always need Newton's unified collision pipeline
            if isinstance(NewtonManager._solver, SolverMuJoCo):
                use_mujoco_contacts = NewtonManager._cfg.solver_cfg.get("use_mujoco_contacts", False)
                NewtonManager._needs_collision_pipeline = not use_mujoco_contacts
            else:
                NewtonManager._needs_collision_pipeline = True

            # Initialize contacts and collision pipeline
            NewtonManager._initialize_contacts()

        # Capture the graph if CUDA is enabled
        with Timer(name="newton_cuda_graph", msg="CUDA graph took:", enable=True, format="ms"):
            if NewtonManager._cfg.use_cuda_graph and NewtonManager._device.startswith("cuda"):
                with wp.ScopedCapture() as capture:
                    NewtonManager.simulate()
                NewtonManager._graph = capture.graph
            elif NewtonManager._cfg.use_cuda_graph and not NewtonManager._device.startswith("cuda"):
                logger.warning("CUDA graphs requested but device is CPU. Disabling CUDA graphs.")
                NewtonManager._cfg.use_cuda_graph = False

    @classmethod
    def simulate(cls) -> None:
        """Simulates the simulation.

        Performs one simulation step with the specified number of substeps. Depending on the solver type, this function
        may need to explicitly compute the collisions. This function also aggregates the contacts and evaluates the
        contact sensors. Finally, it performs the state swapping for Newton.
        """

        # MJWarp can use its internal collision pipeline.
        if cls._needs_collision_pipeline:
            cls._collision_pipeline.collide(cls._state_0, cls._contacts)
            contacts = cls._contacts
        else:
            contacts = None

        def step_fn(state_0, state_1):
            cls._solver.step(state_0, state_1, cls._control, contacts, cls._solver_dt)

        if cls._use_single_state:
            for i in range(cls._num_substeps):
                step_fn(cls._state_0, cls._state_0)
                cls._state_0.clear_forces()
        else:
            need_copy_on_last_substep = cls._cfg.use_cuda_graph and cls._num_substeps % 2 == 1

            for i in range(cls._num_substeps):
                step_fn(cls._state_0, cls._state_1)
                if need_copy_on_last_substep and i == cls._num_substeps - 1:
                    cls._state_0.assign(cls._state_1)
                else:
                    cls._state_0, cls._state_1 = cls._state_1, cls._state_0
                cls._state_0.clear_forces()

        # Transfer contact forces from solver to Newton contacts for sensor evaluation
        if cls._report_contacts:
            # For newton_contacts (unified pipeline): use locally computed contacts
            # For mujoco_contacts: use class-level _contacts, solver populates it from MuJoCo data
            eval_contacts = contacts if contacts is not None else cls._contacts
            cls._solver.update_contacts(eval_contacts, cls._state_0)
            for sensor in cls._newton_contact_sensors.values():
                sensor.eval(eval_contacts)

    @classmethod
    def set_device(cls, device: str) -> None:
        """Sets the device to use for the Newton simulation.

        Args:
            device (str): The device to use for the Newton simulation.
        """
        NewtonManager._device = device

    @classmethod
    def step(cls) -> None:
        """Steps the simulation.

        This function steps the simulation by the specified time step in the simulation configuration.
        """
        if NewtonManager._model_changes:
            for change in NewtonManager._model_changes:
                NewtonManager._solver.notify_model_changed(change)
            NewtonManager._model_changes = set()

        if NewtonManager._cfg.use_cuda_graph:
            wp.capture_launch(NewtonManager._graph)
        else:
            NewtonManager.simulate()

        if NewtonManager._cfg.debug_mode:
            convergence_data = NewtonManager.get_solver_convergence_steps()
            # print(f"solver niter: {convergence_data}")
            if convergence_data["max"] == NewtonManager._solver.mjw_model.opt.iterations:
                print("solver didn't converge!", convergence_data["max"])

        NewtonManager._sim_time += NewtonManager._solver_dt * NewtonManager._num_substeps

    @classmethod
    def get_solver_convergence_steps(cls) -> dict[str, float | int]:
        niter = NewtonManager._solver.mjw_data.solver_niter.numpy()
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
        NewtonManager._dt = dt

    @classmethod
    def get_model(cls):
        return NewtonManager._model

    @classmethod
    def get_state_0(cls):
        return NewtonManager._state_0

    @classmethod
    def get_state_1(cls):
        return NewtonManager._state_1

    @classmethod
    def get_control(cls):
        return NewtonManager._control

    @classmethod
    def get_dt(cls):
        return NewtonManager._dt

    @classmethod
    def get_solver_dt(cls):
        return NewtonManager._solver_dt

    @classmethod
    def forward_kinematics(cls, mask: wp.array | None = None) -> None:
        """Evaluates the forward kinematics for the selected articulations.

        This function evaluates the forward kinematics for the selected articulations.
        """
        eval_fk(
            NewtonManager._model,
            NewtonManager._state_0.joint_q,
            NewtonManager._state_0.joint_qd,
            NewtonManager._state_0,
            None,
        )

    @classmethod
    def _get_solver(cls, model: Model, solver_cfg: dict) -> SolverBase:
        NewtonManager._solver_type = solver_cfg.pop("solver_type")

        if NewtonManager._solver_type == "mujoco_warp":
            # SolverMuJoCo does not require distinct input & output states
            cls._use_single_state = True
            return SolverMuJoCo(model, **solver_cfg)
        elif NewtonManager._solver_type == "xpbd":
            cls._use_single_state = False
            return SolverXPBD(model, **solver_cfg)
        elif NewtonManager._solver_type == "featherstone":
            cls._use_single_state = False
            return SolverFeatherstone(model, **solver_cfg)
        else:
            raise ValueError(f"Invalid solver type: {NewtonManager._solver_type}")

    @classmethod
    def add_contact_sensor(
        cls,
        body_names_expr: str | list[str] | None = None,
        shape_names_expr: str | list[str] | None = None,
        contact_partners_body_expr: str | list[str] | None = None,
        contact_partners_shape_expr: str | list[str] | None = None,
        prune_noncolliding: bool = True,
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

        # Create unique key for this sensor
        sensor_key = (body_names_expr, shape_names_expr, contact_partners_body_expr, contact_partners_shape_expr)

        # Create and store the sensor
        # Note: SensorContact constructor requests 'force' attribute from the model
        newton_sensor = NewtonContactSensor(
            NewtonManager._model,
            sensing_obj_bodies=body_names_expr,
            sensing_obj_shapes=shape_names_expr,
            counterpart_bodies=contact_partners_body_expr,
            counterpart_shapes=contact_partners_shape_expr,
            match_fn=flipped_match,
            include_total=True,
            prune_noncolliding=prune_noncolliding,
            verbose=verbose,
        )
        NewtonManager._newton_contact_sensors[sensor_key] = newton_sensor
        NewtonManager._report_contacts = True

        # Regenerate contacts only if they were already created without force attribute
        # If solver is not initialized, contacts will be created with force in initialize_solver()
        if NewtonManager._solver is not None and NewtonManager._contacts is not None:
            # Only regenerate if contacts don't have force attribute (sensor.eval() requires it)
            if NewtonManager._contacts.force is None:
                NewtonManager._initialize_contacts()

        return sensor_key
