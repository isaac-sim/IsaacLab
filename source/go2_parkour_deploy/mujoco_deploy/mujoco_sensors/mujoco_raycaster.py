import mujoco 
import torch as th 
from isaaclab.utils.math import quat_apply, quat_apply_yaw
from typing import Sequence
from mujoco_deploy.mujoco_sensors.mujoco_base_sensor import MujocoBaseSensor
import numpy as np 
from mujoco.viewer import Handle
from multiprocessing import Process
from dataclasses import dataclass

@dataclass
class RayCasterData:
    pos_w: th.Tensor = None
    quat_w: th.Tensor = None
    ray_hits_w: th.Tensor = None


def grid_pattern(cfg, device: str) -> tuple[th.Tensor, th.Tensor]:
    # check valid arguments
    if cfg.ordering not in ["xy", "yx"]:
        raise ValueError(f"Ordering must be 'xy' or 'yx'. Received: '{cfg.ordering}'.")
    if cfg.resolution <= 0:
        raise ValueError(f"Resolution must be greater than 0. Received: '{cfg.resolution}'.")

    indexing = cfg.ordering if cfg.ordering == "xy" else "ij"
    # define grid pattern
    x = th.arange(start=-cfg.size[0] / 2, end=cfg.size[0] / 2 + 1.0e-9, step=cfg.resolution, device=device)
    y = th.arange(start=-cfg.size[1] / 2, end=cfg.size[1] / 2 + 1.0e-9, step=cfg.resolution, device=device)
    grid_x, grid_y = th.meshgrid(x, y, indexing=indexing)

    # store into ray starts
    num_rays = grid_x.numel()
    ray_starts = th.zeros(num_rays, 3, device=device)
    ray_starts[:, 0] = grid_x.flatten()
    ray_starts[:, 1] = grid_y.flatten()

    # define ray-cast directions
    ray_directions = th.zeros_like(ray_starts)
    ray_directions[..., :] = th.tensor(list(cfg.direction), device=device)

    return ray_starts, ray_directions


def render_sphere(viewer: Handle, position: np.ndarray, diameter: float, color: np.ndarray, geom_id: int = -1) -> int:
    """Function to render a sphere in the Mujoco viewer.

    Args:
        viewer (Handle): The Mujoco viewer.
        position (np.ndarray): The position of the sphere.
        diameter (float): The diameter of the sphere.
        color (np.ndarray): The color of the sphere.
        geom_id (int, optional): The id of the geometry. Defaults to -1.

    Returns:
        int: The id of the geometry.
    """
    if viewer is None:
        return -1

    if geom_id < 0 or geom_id is None:
        # Instantiate a new geometry
        geom = mujoco.MjvGeom()
        geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
        viewer.user_scn.ngeom += 1
        geom_id = viewer.user_scn.ngeom - 1

    geom = viewer.user_scn.geoms[geom_id]

    # Initialize the geometry
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.asarray([diameter / 2] * 3),  # Radius is half the diameter
        mat=np.eye(3).flatten(),
        pos=position,
        rgba=color,
    )

    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    geom.segid = -1
    geom.objid = -1

    return geom_id


class MujocoRaycaster(MujocoBaseSensor):
    def __init__(self, 
                 env_cfg, 
                 articulation,
                 model: mujoco.MjModel, 
                 data:mujoco.MjData 
                 ):
        super().__init__(env_cfg)
        self._model = model
        self._data = data 
        self._ray_cast_data = RayCasterData()
        self._env_cfg = env_cfg
        self.sensor_cfg = env_cfg.scene.height_scanner
        self._articulation = articulation
        self._initialize_impl()

    def _initialize_impl(self):
        super()._initialize_impl()
        self._initialize_rays_impl()

    def _initialize_rays_impl(self):
        self.ray_starts, self.ray_directions = grid_pattern(self.sensor_cfg.pattern_cfg, self._device)
        self.num_rays = len(self.ray_directions)
        offset_pos = th.tensor(list(self.sensor_cfg.offset.pos), device=self._device)
        offset_quat = th.tensor(list(self.sensor_cfg.offset.rot), device=self._device)
        self.ray_starts += offset_pos
        self.ray_directions = quat_apply(offset_quat.repeat(len(self.ray_directions), 1), self.ray_directions)
        self.ray_starts = self.ray_starts.repeat(1, 1, 1)
        self.ray_directions = self.ray_directions.repeat(1, 1, 1)
        self.drift = th.zeros(1, 3, device=self._device)
        self._ray_cast_data.pos_w = th.zeros(1, 3, device=self._device)
        self._ray_cast_data.quat_w = th.zeros(1, 4, device=self._device)
        self._ray_cast_data.ray_hits_w = th.zeros(1, self.num_rays, 3, device=self._device)
        self._geom_ids = -np.ones((self.num_rays), dtype=np.int32)

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timers and counters
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # resample the drift
        self.drift[env_ids] = self.drift[env_ids].uniform_(*self.sensor_cfg.drift_range)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        root_state_w = self._articulation.root_state_w
        pos_w = root_state_w[:,:3].clone().to(dtype=th.float32)
        quat_w = root_state_w[:,3:7].clone().to(dtype=th.float32) 
        # apply drift
        
        pos_w = (pos_w + self.drift[env_ids]).to(dtype=th.float32)        # store the poses
        self._ray_cast_data.pos_w[env_ids] = pos_w
        self._ray_cast_data.quat_w[env_ids] = quat_w
        if self.sensor_cfg.attach_yaw_only:
            # only yaw orientation is considered and directions are not rotated
            ray_starts_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.squeeze(0)
            ray_directions_w = self.ray_directions[env_ids]
        else:
            # full orientation is considered
            ray_starts_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.squeeze(0)
            ray_directions_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])

        geomid = np.zeros(1, np.int32)
        ray_starts_w_numpy = ray_starts_w.detach().cpu().numpy() 
        ray_directions_w_numpy = ray_directions_w.detach().cpu().numpy()
        base_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        for idx,(ray_start, ray_direction) in enumerate(zip(ray_starts_w_numpy[0], ray_directions_w_numpy[0])):
            z = mujoco.mj_ray(
                    m=self._model,
                    d=self._data,
                    pnt=ray_start[:,None],
                    vec=ray_direction[:,None],
                    geomgroup=geomgroup,
                    flg_static=1,
                    bodyexclude=base_body_id,
                    geomid=geomid,
                )
            self._ray_cast_data.ray_hits_w[0, idx, :] = th.from_numpy(ray_start + ray_direction * z).to(device = self._device, dtype = float)

    def render(self, viewer):
        for idx , ray_hits_w in enumerate(self._ray_cast_data.ray_hits_w[0]):
            self._geom_ids[idx] = render_sphere(
                viewer=viewer,
                position=ray_hits_w.detach().cpu().numpy(),
                diameter=0.05,
                color=[0, 1, 0, 0.5],
                geom_id=self._geom_ids[idx],
            )

    @property
    def sensor_data(self):
        self._update_outdated_buffers()
        return self._ray_cast_data


