# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import re

import usdrt
import warp as wp
from isaacsim.core.utils.stage import get_current_stage
from newton import Axis, Contacts, Control, Model, ModelBuilder, State, eval_fk
from newton.sensors import ContactSensor as NewtonContactSensor
from newton.sensors import populate_contacts
from newton.solvers import SolverBase, SolverFeatherstone, SolverMuJoCo, SolverXPBD
from newton.viewer import RendererOpenGL

from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
from isaaclab.utils.timer import Timer


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


@wp.kernel(enable_backward=False)
def set_vec3d_array(
    fabric_vals: wp.fabricarray(dtype=wp.mat44d),
    indices: wp.fabricarray(dtype=wp.uint32),
    newton_vals: wp.array(ndim=1, dtype=wp.transformf),
):
    i = int(wp.tid())
    idx = int(indices[i])
    new_val = newton_vals[idx]
    fabric_vals[i] = wp.transpose(wp.mat44d(wp.math.transform_to_matrix(new_val)))


class NewtonManager:
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
    _newton_contact_sensor: NewtonContactSensor = None  # TODO: allow several contact sensors
    _report_contacts: bool = False
    _graph = None
    _newton_stage_path = None
    _renderer = None
    _sim_time = 0.0
    _usdrt_stage = None
    _newton_index_attr = "newton:index"
    _clone_physics_only = False
    _cfg: NewtonCfg = NewtonCfg()
    _solver_type: str = "mujoco_warp"
    _gravity_vector: tuple[float, float, float] = (0.0, 0.0, -9.81)
    _up_axis: str = "Z"
    _num_envs: int = None

    @classmethod
    def clear(cls):
        NewtonManager._builder = None
        NewtonManager._model = None
        NewtonManager._solver = None
        NewtonManager._state_0 = None
        NewtonManager._state_1 = None
        NewtonManager._state_temp = None
        NewtonManager._control = None
        NewtonManager._contacts = None
        NewtonManager._newton_contact_sensor = None
        NewtonManager._report_contacts = False
        NewtonManager._graph = None
        NewtonManager._newton_stage_path = None
        NewtonManager._renderer = None
        NewtonManager._sim_time = 0.0
        NewtonManager._on_init_callbacks = []
        NewtonManager._on_start_callbacks = []
        NewtonManager._usdrt_stage = None
        NewtonManager._cfg = NewtonCfg()
        NewtonManager._up_axis = "Z"
        NewtonManager._first_call = True

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
    def start_simulation(cls) -> None:
        """Starts the simulation.

        This function finalizes the model and initializes the simulation state.
        """

        print(f"[INFO] Builder: {NewtonManager._builder}")
        if NewtonManager._builder is None:
            NewtonManager.instantiate_builder_from_stage()
        print("[INFO] Running on init callbacks")
        for callback in NewtonManager._on_init_callbacks:
            callback()
        print(f"[INFO] Finalizing model on device: {NewtonManager._device}")
        NewtonManager._builder.gravity = np.array(NewtonManager._gravity_vector)
        NewtonManager._builder.up_axis = Axis.from_string(NewtonManager._up_axis)
        with Timer(name="newton_finalize_builder", msg="Finalize builder took:", enable=True, format="ms"):
            NewtonManager._model = NewtonManager._builder.finalize(device=NewtonManager._device)
        NewtonManager._state_0 = NewtonManager._model.state()
        NewtonManager._state_1 = NewtonManager._model.state()
        NewtonManager._state_temp = NewtonManager._model.state()
        NewtonManager._control = NewtonManager._model.control()
        NewtonManager._contacts = Contacts(0, 0)
        NewtonManager.forward_kinematics()
        print("[INFO] Running on start callbacks")
        for callback in NewtonManager._on_start_callbacks:
            callback()
        if not NewtonManager._clone_physics_only:
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
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        up_axis = UsdGeom.GetStageUpAxis(stage)
        builder = ModelBuilder(up_axis=up_axis)
        builder.add_usd(stage)
        NewtonManager.set_builder(builder)

    @classmethod
    def set_solver_settings(cls, newton_params: dict):
        NewtonManager._cfg = NewtonCfg(**newton_params)

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
        with Timer(name="newton_initialize_solver", msg="Initialize solver took:", enable=True, format="ms"):
            NewtonManager._num_substeps = NewtonManager._cfg.num_substeps
            NewtonManager._solver_dt = NewtonManager._dt / NewtonManager._num_substeps
            NewtonManager._solver = NewtonManager._get_solver(NewtonManager._model, NewtonManager._cfg.solver_cfg)

        # Ensure we are using a CUDA enabled device
        assert NewtonManager._device.startswith("cuda"), "NewtonManager only supports CUDA enabled devices"

        # Capture the graph if CUDA is enabled
        with Timer(name="newton_cuda_graph", msg="CUDA graph took:", enable=True, format="ms"):
            if NewtonManager._cfg.use_cuda_graph:
                with wp.ScopedCapture() as capture:
                    NewtonManager.simulate()
                NewtonManager._graph = capture.graph

    @classmethod
    def simulate(cls) -> None:
        """Simulates the simulation.

        Performs one simulation step with the specified number of substeps. Depending on the solver type, this function
        may need to explicitly compute the collisions. This function also aggregates the contacts and evaluates the
        contact sensors. Finally, it performs the state swapping for Newton.
        """
        state_0_dict = NewtonManager._state_0.__dict__
        state_1_dict = NewtonManager._state_1.__dict__
        state_temp_dict = NewtonManager._state_temp.__dict__
        contacts = None

        # MJWarp computes its own collisions.
        if NewtonManager._solver_type != "mujoco_warp":
            contacts = NewtonManager._model.collide(NewtonManager._state_0)

        if NewtonManager._num_substeps % 2 == 0:
            for i in range(NewtonManager._num_substeps):
                NewtonManager._solver.step(
                    NewtonManager._state_0,
                    NewtonManager._state_1,
                    NewtonManager._control,
                    contacts,
                    NewtonManager._solver_dt,
                )
                NewtonManager._state_0, NewtonManager._state_1 = NewtonManager._state_1, NewtonManager._state_0
                NewtonManager._state_0.clear_forces()
        else:
            for i in range(NewtonManager._num_substeps):
                NewtonManager._solver.step(
                    NewtonManager._state_0,
                    NewtonManager._state_1,
                    NewtonManager._control,
                    contacts,
                    NewtonManager._solver_dt,
                )

                # FIXME: Ask Lukasz help to deal with non-even number of substeps. This should not be needed.
                if i < NewtonManager._num_substeps - 1 or not NewtonManager._cfg.use_cuda_graph:
                    # we can just swap the state references
                    NewtonManager._state_0, NewtonManager._state_1 = NewtonManager._state_1, NewtonManager._state_0
                elif NewtonManager._cfg.use_cuda_graph:
                    # swap states by actually copying the state arrays to make sure the graph capture works
                    for key, value in state_0_dict.items():
                        if isinstance(value, wp.array):
                            if key not in state_temp_dict:
                                state_temp_dict[key] = wp.empty_like(value)
                            state_temp_dict[key].assign(value)
                            state_0_dict[key].assign(state_1_dict[key])
                            state_1_dict[key].assign(state_temp_dict[key])
                NewtonManager._state_0.clear_forces()

        if NewtonManager._report_contacts:
            populate_contacts(NewtonManager._contacts, NewtonManager._solver)
            NewtonManager._newton_contact_sensor.eval(NewtonManager._contacts)

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
        if NewtonManager._cfg.use_cuda_graph:
            wp.capture_launch(NewtonManager._graph)
        else:
            NewtonManager.simulate()

        if NewtonManager._cfg.debug_mode:
            convergence_data = NewtonManager.get_solver_convergence_steps()
            print(f"solver niter: {convergence_data}")
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
    def render(cls) -> None:
        """Renders the simulation.

        This function renders the simulation using the OpenGL renderer.
        """
        if NewtonManager._renderer is None:
            NewtonManager._renderer = RendererOpenGL(
                path="example.usd",
                model=NewtonManager._model,
                scaling=1.0,
                up_axis=NewtonManager._up_axis,
                screen_width=1280,
                screen_height=720,
                camera_pos=(0, 3, 10),
            )
        else:
            NewtonManager._renderer.begin_frame(NewtonManager._sim_time)
            NewtonManager._renderer.render(NewtonManager._state_0)
            NewtonManager._renderer.end_frame()

    @classmethod
    def sync_fabric_transforms(cls) -> None:
        """Syncs the fabric transforms with the Newton state.

        This function syncs the fabric transforms with the Newton state.
        """
        selection = NewtonManager._usdrt_stage.SelectPrims(
            require_attrs=[
                (usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:worldMatrix", usdrt.Usd.Access.ReadWrite),
                (usdrt.Sdf.ValueTypeNames.UInt, NewtonManager._newton_index_attr, usdrt.Usd.Access.Read),
            ],
            device="cuda:0",
        )
        fabric_newton_indices = wp.fabricarray(selection, NewtonManager._newton_index_attr)
        current_transforms = wp.fabricarray(selection, "omni:fabric:worldMatrix")
        wp.launch(
            set_vec3d_array,
            dim=(fabric_newton_indices.shape[0]),
            inputs=[current_transforms, fabric_newton_indices, NewtonManager._state_0.body_q],
            device="cuda:0",
        )

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
            return SolverMuJoCo(model, **solver_cfg)
        elif NewtonManager._solver_type == "xpbd":
            return SolverXPBD(model, **solver_cfg)
        elif NewtonManager._solver_type == "featherstone":
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

        NewtonManager._newton_contact_sensor = NewtonContactSensor(
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
        NewtonManager._report_contacts = True
