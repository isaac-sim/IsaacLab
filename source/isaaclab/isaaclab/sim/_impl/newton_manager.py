import newton.sim.articulation
import warp as wp
import newton.utils
#from newton.utils.selection import ContactViewManager, ContactView
from isaacsim.core.utils.stage import print_stage_prim_paths, get_current_stage
from newton import Model, State, Control
from newton.sim import ModelBuilder
import usdrt

@wp.kernel(enable_backward=False)
def set_vec3d_array(
    fabric_vals: wp.fabricarray(dtype=wp.mat44d),
    idices: wp.fabricarray(dtype=wp.uint32),
    newton_vals: wp.array(ndim=1, dtype=wp.transformf)
):
    i = int(wp.tid())
    idx = int(idices[i])
    new_val = newton_vals[idx]
    fabric_vals[i] = wp.transpose(wp.mat44d(wp.math.transform_to_matrix(new_val)))

class NewtonManager:
    _builder: ModelBuilder = None
    #_contact_manager: ContactViewManager = None
    _model: Model = None
    _device: str = "cuda:0"
    _sim_dt: float = 1.0 / 600.0
    _decimation: int = 10
    _solver = None
    _state_0: State = None
    _state_1: State = None
    _state_temp: State = None
    _control: Control = None
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
    def start_simulation(cls):
        NewtonManager._model = NewtonManager._builder.finalize(device=NewtonManager._device)
        #NewtonManager._contact_manager = ContactViewManager(NewtonManager._model)
        NewtonManager._model.ground = True    
        NewtonManager._state_0 = NewtonManager._model.state()
        NewtonManager._state_1 = NewtonManager._model.state()
        NewtonManager._state_temp = NewtonManager._model.state()
        NewtonManager._control = NewtonManager._model.control()
        newton.sim.articulation.eval_fk(NewtonManager._model, NewtonManager._model.joint_q, NewtonManager._model.joint_qd, NewtonManager._state_0, None)
        NewtonManager._usdrt_stage = get_current_stage(fabric=True)
        for i, prim_path in enumerate(NewtonManager._model.body_key):
            #print("Being added to fabric: ", prim_path)
            prim = NewtonManager._usdrt_stage.GetPrimAtPath(prim_path)
            prim.CreateAttribute(NewtonManager._newton_index_attr, usdrt.Sdf.ValueTypeNames.UInt, True)
            prim.GetAttribute(NewtonManager._newton_index_attr).Set(i)
            xformable_prim = usdrt.Rt.Xformable(prim)
            if not xformable_prim.HasWorldXform():
                xformable_prim.SetWorldXformFromUsd()

    @classmethod
    def initialize_solver(cls):
        NewtonManager._solver = newton.solvers.MuJoCoSolver(NewtonManager._model)
        NewtonManager._use_cuda_graph = wp.get_device().is_cuda
        #NewtonManager._contact_manager.finalize(NewtonManager._solver)
        if NewtonManager._use_cuda_graph:
            with wp.ScopedCapture() as capture:
                NewtonManager.simulate()
            NewtonManager._graph = capture.graph

    @classmethod
    def simulate(cls):
        state_0_dict = NewtonManager._state_0.__dict__
        state_1_dict = NewtonManager._state_1.__dict__
        state_temp_dict = NewtonManager._state_temp.__dict__
        for i in range(NewtonManager._decimation):
            NewtonManager._state_0.clear_forces()
            NewtonManager._solver.step(NewtonManager._model, NewtonManager._state_0, NewtonManager._state_1, NewtonManager._control, None, NewtonManager._sim_dt)
            if i < NewtonManager._decimation - 1 or not NewtonManager._use_cuda_graph:
                # we can just swap the state references
                NewtonManager._state_0, NewtonManager._state_1 = NewtonManager._state_1, NewtonManager._state_0
            elif  NewtonManager._use_cuda_graph:
                # swap states by actually copying the state arrays to make sure the graph capture works
                for key, value in state_0_dict.items():
                    if isinstance(value, wp.array):
                        if key not in state_temp_dict:
                            state_temp_dict[key] = wp.empty_like(value)
                        state_temp_dict[key].assign(value)
                        state_0_dict[key].assign(state_1_dict[key])
                        state_1_dict[key].assign(state_temp_dict[key])
        #with wp.ScopedTimer("Contacts aggregation"):
        #    contact = NewtonManager._model.contact()
        #    contact.dist = NewtonManager._solver.mjw_data.contact.dist
        #    contact.geom = NewtonManager._solver.mjw_data.contact.geom
        #    contact.frame = NewtonManager._solver.mjw_data.contact.frame
        #    contact.worldid = NewtonManager._solver.mjw_data.contact.worldid

        #    n_contacts = NewtonManager._solver.mjw_data.ncon
        #    NewtonManager._contact_manager.contact_reporter.select_aggregate(contact, n_contacts)

    @classmethod
    def set_device(cls, device: str) -> None:
        """Sets the device to use for the Newton simulation.

        Args:
            device (str): The device to use for the Newton simulation.
        """
        NewtonManager._device = device

    @classmethod
    def step(cls):
        with wp.ScopedTimer("step", active=False):
            if NewtonManager._use_cuda_graph:
                wp.capture_launch(NewtonManager._graph)
            else:
                NewtonManager.simulate()
        NewtonManager._sim_time += (NewtonManager._sim_dt * NewtonManager._decimation)

    @classmethod
    def set_simulation_dt(cls, sim_dt, decimation):
        NewtonManager._sim_dt = sim_dt
        NewtonManager._decimation = decimation


    @classmethod
    def render(cls):
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
    def sync_fabric_transforms(cls): 
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
    def get_control(cls):
        return NewtonManager._control


    @classmethod
    def forward_kinematics(cls, selection):
        newton.sim.articulation.eval_fk(
            NewtonManager._model, NewtonManager._state_0.joint_q, NewtonManager._state_0.joint_qd, NewtonManager._state_0, selection.articulation_mask 
        )

    #@classmethod
    #def add_contact_view(cls, body_names_glob, filter_prim_paths_glob):
    #    print(f"[INFO] Adding contact view for {body_names_glob} with filter {filter_prim_paths_glob}")
    #    return ContactView(NewtonManager._contact_manager, body_names_glob, filter_prim_paths_glob)

    #@classmethod
    #def get_contact_view(cls):
    #    return NewtonManager._contact_manager