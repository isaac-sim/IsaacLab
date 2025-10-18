import mujoco
import mujoco.viewer 
import numpy as np 
import torch as th 
from core.utils import isaac_to_mujoco, ISAAC_JOINT_NAMES
import re, time
from mujoco_deploy.mujoco_sensors.mujoco_articulation import MujocoArticulation

class MujocoEnv():
    """
    Mujoco Environment
    """
    def __init__(
        self, 
        env_cfg,
        model_xml_path,
        use_camera    
        ):
        self.env_cfg = env_cfg
        self.model_xml_path = model_xml_path
        self.use_camera = use_camera
        self._init_mujoco()
        self._init_pose()
        self._reset_counter = 0

    def _init_pose(self):
        init_state = self.env_cfg.scene.robot.init_state
        default_joint_compiled = {re.compile(joint_name): value  for joint_name, value in init_state.joint_pos.items()}
        default_joint_data = np.zeros((1,len(ISAAC_JOINT_NAMES)))
        for joint_name_compiled in default_joint_compiled.keys():
            for indx,real_joint_name in enumerate(ISAAC_JOINT_NAMES):
                if joint_name_compiled.match(real_joint_name):
                    default_joint_data[0,indx] = default_joint_compiled[joint_name_compiled]

        self._nominal_joint_pos = np.hstack(
                            [
                            np.array(init_state.pos).squeeze(), 
                            np.array(init_state.rot).squeeze(), 
                            np.array(default_joint_data).squeeze()
                            ]
                            )[None, 7:]
        self._default_joint_vel = np.hstack(
                            [
                            np.array(init_state.pos).squeeze(), 
                            np.array(init_state.rot).squeeze(), 
                            np.array(init_state.lin_vel).squeeze(),
                            np.array(init_state.ang_vel).squeeze()]
                            )[None]
        self._default_joint_pose = th.zeros((1, 
                                            self._articulation.num_motor),
                                            device = self._articulation.device)
    def _init_mujoco(self):
        self._model = mujoco.MjModel.from_xml_path(self.model_xml_path)
        self._data = mujoco.MjData(self._model)
        bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

        self.viewer = mujoco.viewer.launch_passive(
                                                   self._model, 
                                                   self._data
                                                   )
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.viewer.cam.trackbodyid = bid

        self.viewer.cam.distance = 3.0  
        self.viewer.cam.elevation = -30 
        self.viewer.cam.azimuth = 90    
        self._model.opt.timestep = self.env_cfg.sim.dt 
        self._model.opt.gravity = np.array([self.env_cfg.sim.gravity])
        self._articulation = MujocoArticulation(self.env_cfg, self._model, self._data)

    def reset(self):
        mujoco.mj_resetData(self._model, self._data)
        self._default_joint_pose[:] = th.tensor(self._nominal_joint_pos).to(self._articulation.device)
        self._articulation.joint_vel = th.from_numpy(self._default_joint_vel[:,1:]).to(self._articulation.device)

    def step(self, actions: np.ndarray | None = None) -> None:
        step_start = time.perf_counter()
        if actions is None:
            actions = np.zeros((1, self._model.nu))

        if actions.shape != (1, self._model.nu):
            raise ValueError(
                f"Action shape {actions.shape} does not match number of actuators"
                f" {(1, self._model.nu)}"
            )
        for idx, joint_idx in enumerate(isaac_to_mujoco):
            self._data.ctrl[idx] = actions[0][joint_idx]
            
        self.articulation.update(dt = self.env_cfg.sim.dt )
        time_until_next_step = self._model.opt.timestep - (
            time.perf_counter() - step_start
        )
        mujoco.mj_step(self._model, self._data)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        self.viewer.sync()

    @property
    def articulation(self):
        return self._articulation
    
    @property
    def data(self):
        return self._data
    
    @property
    def model(self):
        return self._model
    
    @property
    def default_joint_vel(self):
        return th.from_numpy(self._default_joint_vel[:,1:]).to(self._articulation.device)
    

    @default_joint_vel.setter 
    def default_joint_vel(self, value: th.Tensor):
        self._default_joint_vel[:] = value


    @property 
    def default_joint_pose(self):
        return self._default_joint_pose

    @default_joint_pose.setter 
    def default_joint_pose(self, value: th.Tensor):
        self._default_joint_pose[:] = value

    def close(self):
        self.viewer.close()


