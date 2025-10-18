import mujoco
import isaaclab.utils.math as math_utils
import torch as th 
from isaaclab.utils.buffers import TimestampedBuffer
from core.utils import get_entity_name, get_entity_id, ISAAC_JOINT_NAMES, mujoco_to_isaac, isaac_to_mujoco
import numpy as np 
from typing import Tuple
import re 
from scipy.spatial.transform import Rotation

class MujocoArticulation():
    """
    Wrapping Mujoco data into IsaacLab style
    """
    def __init__(self, env_cfg ,model: mujoco.MjModel, data:mujoco.MjData ):
        self._model = model
        self._data = data 
        self._env_cfg = env_cfg
        actuator_consistency = self._check_actuator_consistency()
        assert actuator_consistency, "Only support that all the actuator use the same control type."
        self._sim_timestamp = 0.0
        self._num_motor = self._model.nu
        self._dim_motor_sensor = 3 * self._num_motor
        self._has_free_joint = self._model.nv != self._model.nq
        self.joint_pos_offset = self._model.nq - self._model.nu  # Because positions are in generalized coordinates
        self.joint_vel_offset = self._model.nv - self._model.nu  # Because velocities include the free joint
        self._device = 'cuda' if th.cuda.is_available() else 'cpu'
        self._init_buffer()
        self._init_joint_info()

    def _init_joint_info(self):
        self._joint_stiffness = th.ones(self._num_motor).to(self._device)
        self._joint_dampings = th.ones(self._num_motor).to(self._device)
        self._joint_stiffness *= self._env_cfg.scene.robot.actuators.base_legs.stiffness
        self._joint_dampings *= self._env_cfg.scene.robot.actuators.base_legs.damping
        self._body_com = np.zeros(3)
        self._total_mass = 0.0
        base_link_id = self.get_body_ids(['base_link'])['base_link']
        self._body_mass = self._model.body_mass[base_link_id]
        self._com = self._data.subtree_com[base_link_id]
        self._body_com += self._body_mass * self._com
        self._total_mass += self._body_mass
        self._body_com = th.from_numpy(self._body_com).to(self._device)/ self._total_mass
        self._body_mass = th.tensor([self._total_mass]).to(self._device)
        self._joint_efforts = th.zeros((1, self._num_motor), dtype=th.float, device=self.device)
        self._control_joint_velocities = th.zeros((1, self._num_motor), dtype=th.float, device=self.device)
        self._zeros_effort = th.zeros((1, self._num_motor), device=self.device)

        saturation_limit_compiled = {re.compile(joint_name): value  for joint_name, value in self._env_cfg.scene.robot.actuators.base_legs.saturation_effort.items()}
        velocity_limit_compiled = {re.compile(joint_name): value  for joint_name, value in self._env_cfg.scene.robot.actuators.base_legs.velocity_limit.items()}
        effort_limit_compiled = {re.compile(joint_name): value  for joint_name, value in self._env_cfg.scene.robot.actuators.base_legs.effort_limit.items()}
        self._saturation_effort = th.zeros((1, self._num_motor), device=self.device)
        self._velocity_limit = th.zeros((1, self._num_motor), device=self.device)
        self._effort_limit = th.zeros((1, self._num_motor), device=self.device)

        for joint_name_compiled in saturation_limit_compiled.keys():
            for indx,real_joint_name in enumerate(ISAAC_JOINT_NAMES):
                if joint_name_compiled.match(real_joint_name):
                    self._saturation_effort[0,indx] = saturation_limit_compiled[joint_name_compiled]
                    self._velocity_limit[0,indx] = velocity_limit_compiled[joint_name_compiled]
                    self._effort_limit[0,indx] = effort_limit_compiled[joint_name_compiled]

    @property
    def saturation_effort(self):
        return self._saturation_effort
    
    @property
    def velocity_limit(self):
        return self._velocity_limit

    @property
    def control_joint_velocities(self):
        return self._control_joint_velocities

    @property
    def zeros_effort(self):
        return self._zeros_effort

    @property
    def effort_limit(self):
        return self._effort_limit

    @property
    def joint_efforts(self):
        return self._joint_efforts

    @joint_efforts.setter
    def joint_efforts(self, value: th.Tensor):
        self._joint_efforts[:] = value

    @property
    def body_com(self):
        return self._body_com

    @property
    def body_mass(self):
        return self._body_mass

    @property
    def joint_stiffness(self):
        return self._joint_stiffness

    @property
    def joint_dampings(self):
        return self._joint_dampings

    @property
    def body_names(self) -> list[str]:
        return [get_entity_name(self._model, "body", i) for i in range(1, self._model.nbody)]

    def get_body_ids(self, body_names: list[str] | None = None, free_joint_offset: int = 1) -> dict[str, int]:
        body_names_ = body_names if body_names else self.body_names
        body_ids = {}
        for name in body_names_:
            id_ = get_entity_id(self._model, "body", name)
            if id_ > 0:
                body_ids[name] = id_ - free_joint_offset
            else:
                body_ids[name] = id_
        return body_ids
    
    @property
    def joint_names(self) -> list[str]:
        offset = 0
        if self._has_free_joint:
            offset = 1  
        return [get_entity_name(self._model, "joint", i) for i in range(offset, self._model.njnt)]

    def get_joint_ids(self, joint_names: list[str] | None = None, free_joint_offset: int = 1) -> dict[str, int]:
        joint_name_ = joint_names if joint_names else self.joint_names
        joint_ids = {}
        for name in joint_name_:
            id_ = get_entity_id(self._model, "joint", name)
            if id_ > 0:
                joint_ids[name] = id_ - free_joint_offset
            else:
                joint_ids[name] = id_
        return joint_ids
    
    def _check_actuator_consistency(self):
        """Check whether all the actuators share the same control mode."""
        actuator_type_system = None
        for actuator_id in range(self._model.nu):
            actuator_type = self._model.actuator_trntype[actuator_id]
            if actuator_type_system is None:
                actuator_type_system = actuator_type
            else:
                if actuator_type_system != actuator_type:
                    return False
        return True

    def update(self, dt: float):
        self._sim_timestamp += dt

    def _init_buffer(self):
        self._root_state_w = TimestampedBuffer()
        self._body_state_w = TimestampedBuffer()
        self._joint_pos = TimestampedBuffer()
        self._joint_vel = TimestampedBuffer()

        
    @property
    def root_state_w(self):
        bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        if bid < 0:
            raise ValueError(f"Body 'base_link' not found")
        pos_w   = th.from_numpy(self._data.xpos[bid].copy()).to(device=self._device)        
        quat_w  = th.from_numpy(self._data.xquat[bid].copy()).to(device=self._device)         

        cvel_local = self._data.cvel[bid].copy()                                 
        R = self._data.xmat[bid].reshape(3, 3).copy()                             
        ang_w = R @ cvel_local[0:3]                                    
        lin_w = R @ cvel_local[3:6]                                 

        ang_w = th.from_numpy(ang_w).to(device=self._device)
        lin_w = th.from_numpy(lin_w).to(device=self._device)

        root_state = th.cat([pos_w, quat_w, lin_w, ang_w], dim=0).unsqueeze(0)
        return root_state
    

    @property
    def joint_pos(self):
        if self._joint_pos.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._joint_pos.data = th.from_numpy(self._data.sensordata[:self._num_motor].copy())\
                                    .to(dtype=th.float32, device=self._device).expand(1, -1)
            self._joint_pos.timestamp = self._sim_timestamp
        return self._joint_pos.data[:,mujoco_to_isaac] 

    @property
    def joint_vel(self):
        if self._joint_vel.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._joint_vel.data = th.from_numpy(self._data.sensordata[self._num_motor:self._num_motor+self._num_motor].copy())\
                                    .to(dtype=th.float32, device=self._device).expand(1, -1)
            self._joint_vel.timestamp = self._sim_timestamp
        return self._joint_vel.data[:,mujoco_to_isaac]

    @property
    def root_quat_w(self) -> th.Tensor:
        return self.root_state_w[:, 3:7]

    @property
    def root_ang_vel_w(self) ->th.Tensor:
        return self.root_state_w[:, 10:13]

    @property
    def root_ang_vel_b(self) -> th.Tensor:
        return math_utils.quat_rotate_inverse(self.root_quat_w, self.root_ang_vel_w)

    @property
    def device(self):
        return self._device
    
    @property 
    def num_motor(self):
        return self._num_motor
    
    @joint_vel.setter
    def joint_vel(self, value: th.Tensor):
        assert value.shape[-1] == self._num_motor, \
            f"joint_vel must have shape (1, {self._num_motor}), but got {value.shape}"
        value_np = value.squeeze(0).detach().cpu().numpy()
        self._data.sensordata[:self._num_motor] = value_np
        self._joint_vel.data = value.clone().detach()
        self._joint_vel.timestamp = self._sim_timestamp