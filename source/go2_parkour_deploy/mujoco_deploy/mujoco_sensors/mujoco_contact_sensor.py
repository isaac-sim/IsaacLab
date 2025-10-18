from typing import Sequence
from .mujoco_base_sensor import MujocoBaseSensor 
import mujoco
import numpy as np
import torch as th 
from dataclasses import dataclass
@dataclass
class ContactSensorData:
    net_forces_w: th.Tensor | None = None
    net_forces_w_history: th.Tensor | None = None
    foot_ids: th.Tensor | None = None

class MujocoContactSensor(MujocoBaseSensor):

    def __init__(self, 
                 env_cfg, 
                 articulation,
                 model:mujoco.MjModel, 
                 data:mujoco.MjData 
                 ):
        super().__init__(env_cfg)
        self._model = model
        self._articulation = articulation
        body_id = self._articulation.get_body_ids()
        self._num_bodies = len(body_id)
        self._data = data
        self._env_cfg = env_cfg
        self.sensor_cfg = env_cfg.scene.contact_forces
        self._contact_data = ContactSensorData()
        self._initialize_impl()


    def _initialize_impl(self):
        super()._initialize_impl()
        self._contact_data.net_forces_w = th.zeros(1, self._num_bodies, 3, device=self._device)
        # optional buffers
        # -- history of net forces
        if self.sensor_cfg.history_length > 0:
            self._contact_data.net_forces_w_history = th.zeros(
                1, self.sensor_cfg.history_length, self._num_bodies, 3, device=self._device
            )
        else:
            self._contact_data.net_forces_w_history = self._contact_data.net_forces_w.unsqueeze(1)
        body_dict = self._articulation.get_body_ids(['FL_foot', 'FR_foot' ,'RL_foot','RR_foot'])
        self._contact_data.foot_ids = th.tensor([_id for _id in body_dict.values()]).to(self._device)

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        self._contact_data.net_forces_w[env_ids] = 0.0
        self._contact_data.net_forces_w_history[env_ids] = 0.0
        if self.sensor_cfg.history_length > 0:
            self._contact_data.net_forces_w_history[env_ids] = 0.0

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        body_ids = [body_id for key, body_id in self._articulation.get_body_ids().items()]
        zero_contact = th.zeros(1, self._num_bodies, 3).to(dtype=th.float32, device=self._device)
        for i in range(self._data.ncon):    
            contact = self._data.contact[i]
            body1_id = self._model.geom_bodyid[contact.geom1]
            body2_id = self._model.geom_bodyid[contact.geom2]
            if 0 in [body1_id, body2_id]:  # World body ID is 0
                second_id = body2_id if body1_id == 0 else body1_id
                if second_id in body_ids:  # Check if contact occurs with the feet
                    for foot_name in ['FL_foot', 'FR_foot' ,'RL_foot','RR_foot']:
                        contact_force = np.zeros(6)
                        mujoco.mj_contactForce(self._model, self._data, i, contact_force)
                        zero_contact[0,second_id,:] = th.from_numpy(contact_force[:3].copy()).to(device=self._device).expand(1, -1)

        self._contact_data.net_forces_w[env_ids, :, :] = zero_contact.view(-1, self._num_bodies, 3)[env_ids]
        if self.sensor_cfg.history_length > 0:
            self._contact_data.net_forces_w_history[env_ids, 1:] = self._contact_data.net_forces_w_history[env_ids, :-1].clone()
            self._contact_data.net_forces_w_history[env_ids, 0] = self._contact_data.net_forces_w[env_ids]

    @property
    def sensor_data(self):
        self._update_outdated_buffers()
        return self._contact_data


