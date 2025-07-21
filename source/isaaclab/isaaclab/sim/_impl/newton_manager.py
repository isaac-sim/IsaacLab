# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import re

import newton.sim.articulation
import newton.utils
import usdrt
import warp as wp
from isaacsim.core.utils.stage import get_current_stage
from newton import Control, Model, State
from newton.sim import ModelBuilder
from newton.sim.contacts import ContactInfo
from newton.utils.contact_sensor import ContactView, convert_contact_info


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
    _sim_dt: float = 1.0 / 200.0
    _solver_type: str = "mjwarp"  # "xpbd, mjwarp
    _num_substeps: int = 1
    _solver = None
    _state_0: State = None
    _state_1: State = None
    _state_temp: State = None
    _control: Control = None
    _on_init_callbacks: list = []
    _on_start_callbacks: list = []
    _contact_info: ContactInfo = None
    _report_contacts: bool = False
    _use_cuda_graph: bool = False
    _graph = None
    _up_axis: str = "Z"
    _newton_stage_path = None
    _renderer = None
    _sim_time = 0.0
    _usdrt_stage = None
    _newton_index_attr = "newton:index"
    _env_offsets = None

    @property
    def model(self) -> Model:
        return NewtonManager._model

    @property
    def state_0(self) -> State:
        return NewtonManager._state_0

    @property
    def state_1(self) -> State:
        return NewtonManager._state_1

    @property
    def control(self) -> Control:
        return NewtonManager._control

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
        print("[INFO] Running on init callbacks")
        for callback in NewtonManager._on_init_callbacks:
            callback()
        print(f"[INFO] Finalizing model on device: {NewtonManager._device}")
        NewtonManager._model = NewtonManager._builder.finalize(device=NewtonManager._device)
        print("[INFO] Running on start callbacks")
        for callback in NewtonManager._on_start_callbacks:
            callback()
        NewtonManager._state_0 = NewtonManager._model.state()
        NewtonManager._state_1 = NewtonManager._model.state()
        NewtonManager._state_temp = NewtonManager._model.state()
        NewtonManager._control = NewtonManager._model.control()
        NewtonManager._contact_info = ContactInfo()
        newton.sim.articulation.eval_fk(
            NewtonManager._model,
            NewtonManager._model.joint_q,
            NewtonManager._model.joint_qd,
            NewtonManager._state_0,
            None,
        )
        NewtonManager._usdrt_stage = get_current_stage(fabric=True)
        for i, prim_path in enumerate(NewtonManager._model.body_key):
            prim = NewtonManager._usdrt_stage.GetPrimAtPath(prim_path)
            prim.CreateAttribute(NewtonManager._newton_index_attr, usdrt.Sdf.ValueTypeNames.UInt, True)
            prim.GetAttribute(NewtonManager._newton_index_attr).Set(i)
            xformable_prim = usdrt.Rt.Xformable(prim)
            if not xformable_prim.HasWorldXform():
                xformable_prim.SetWorldXformFromUsd()

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
        if NewtonManager._solver_type == "xpbd":
            NewtonManager._solver = newton.solvers.XPBDSolver(NewtonManager._model, iterations=20)
        elif NewtonManager._solver_type == "mjwarp":
            NewtonManager._solver = newton.solvers.MuJoCoSolver(
                NewtonManager._model,
                solver="newton",
                ls_iterations=50,
                iterations=100,
                ncon_per_env=300,
                contact_stiffness_time_const=0.01,
                impratio=100,
                cone="elliptic",
            )
        else:
            raise ValueError(f"Unknown solver type: {NewtonManager._solver_type}")

        NewtonManager._use_cuda_graph = wp.get_device().is_cuda
        if NewtonManager._use_cuda_graph:
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
        if NewtonManager._solver_type != "mjwarp":
            contacts = NewtonManager._model.collide(NewtonManager._state_0)

        for i in range(NewtonManager._num_substeps):
            NewtonManager._solver.step(
                NewtonManager._state_0, NewtonManager._state_1, NewtonManager._control, contacts, NewtonManager._sim_dt
            )

            # FIXME: Ask Lukasz help to deal with non-even number of substeps. This should not be needed.
            if i < NewtonManager._num_substeps - 1 or not NewtonManager._use_cuda_graph:
                # we can just swap the state references
                NewtonManager._state_0, NewtonManager._state_1 = NewtonManager._state_1, NewtonManager._state_0
            elif NewtonManager._use_cuda_graph:
                # swap states by actually copying the state arrays to make sure the graph capture works
                for key, value in state_0_dict.items():
                    if isinstance(value, wp.array):
                        if key not in state_temp_dict:
                            state_temp_dict[key] = wp.empty_like(value)
                        state_temp_dict[key].assign(value)
                        state_0_dict[key].assign(state_1_dict[key])
                        state_1_dict[key].assign(state_temp_dict[key])

        if NewtonManager._report_contacts:
            convert_contact_info(NewtonManager._model, NewtonManager._contact_info, NewtonManager._solver)
            NewtonManager._model.eval_contact_sensors(NewtonManager._contact_info)
        NewtonManager._state_0.clear_forces()

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
        with wp.ScopedTimer("step", active=False):
            if NewtonManager._use_cuda_graph:
                wp.capture_launch(NewtonManager._graph)
            else:
                NewtonManager.simulate()

        NewtonManager._sim_time += NewtonManager._sim_dt * NewtonManager._num_substeps

    @classmethod
    def set_simulation_dt(cls, sim_dt: float, substeps: int | None = None) -> None:
        """Sets the simulation time step and the number of substeps.

        Args:
            sim_dt (float): The simulation time step.
            substeps (int | None): The number of substeps. If None, the number of substeps will be set to the default value.
        """
        if substeps is not None:
            NewtonManager._num_substeps = substeps
        NewtonManager._sim_dt = sim_dt / NewtonManager._num_substeps

    @classmethod
    def render(cls) -> None:
        """Renders the simulation.

        This function renders the simulation using the OpenGL renderer.
        """
        if NewtonManager._renderer is None:
            NewtonManager._renderer = newton.utils.SimRendererOpenGL(
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
    def forward_kinematics(cls, selection) -> None:
        """Evaluates the forward kinematics for the selected articulations.

        This function evaluates the forward kinematics for the selected articulations.
        """
        newton.sim.articulation.eval_fk(
            NewtonManager._model,
            NewtonManager._state_0.joint_q,
            NewtonManager._state_0.joint_qd,
            NewtonManager._state_0,
            selection.articulation_mask,
        )

    @classmethod
    def add_contact_view(
        cls,
        il_contact_sensor,
        body_names_expr: str | None = None,
        shape_names_expr: str | None = None,
        contact_partners_body_expr: str | None = None,
        contact_partners_shape_expr: str | None = None,
        include_total: bool = True,
        verbose: bool = False,
    ) -> ContactView:
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
            verbose (bool): Whether to print verbose information.

        Returns:
            ContactView: The contact view.
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

        contact_sensor = NewtonManager._builder.add_contact_sensor(
            sensor_body=body_names_expr,
            sensor_shape=shape_names_expr,
            contact_partners_body=contact_partners_body_expr,
            contact_partners_shape=contact_partners_shape_expr,
            match_fun=flipped_match,
            include_total=include_total,
            verbose=verbose,
        )
        NewtonManager._report_contacts = True
        return contact_sensor
