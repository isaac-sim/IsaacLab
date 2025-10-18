import mujoco 
import torch as th 
from typing import Sequence, Any
from mujoco_deploy.mujoco_sensors.mujoco_base_sensor import MujocoBaseSensor
import numpy as np 
from dataclasses import dataclass
import mujoco, cv2
import os
import matplotlib.pyplot as plt

@dataclass
class CameraData:
    image_shape: tuple[int, int] = None
    intrinsic_matrices: th.Tensor = None
    output: dict[str, th.Tensor] = None
    info: list[dict[str, Any]] = None

class MujocoDepthCamera(MujocoBaseSensor):
    def __init__(self, 
                 env_cfg, 
                 device,
                 model:mujoco.MjModel, 
                 data:mujoco.MjData 
                 ):
        super().__init__(env_cfg)
        self._camera_data = CameraData()
        self._cam_name = 'd435i_camera' 
        self._device = device
        self.sensor_cfg = env_cfg.scene.depth_camera
        self._data = data
        self._model = model
        self._imshow_initialized = False
        self._imshow_handle = None
        self._initialize_impl()

    def _initialize_impl(self):
        super()._initialize_impl()
        self._ALL_INDICES = th.arange(1, device=self._device, dtype=th.long)

        self._create_buffers()
        self._update_intrinsic_matrices()

        self._renderer = mujoco.Renderer(self._model, self.sensor_cfg.pattern_cfg.height, self.sensor_cfg.pattern_cfg.width)
        self._scene = mujoco.MjvScene(self._model, maxgeom=10_000)
        self._image = np.zeros((self.sensor_cfg.pattern_cfg.height, self.sensor_cfg.pattern_cfg.width, 3), dtype=np.uint8)
        self._depth_image = np.zeros((self.sensor_cfg.pattern_cfg.height, self.sensor_cfg.pattern_cfg.width, 1), dtype=np.float32)

    def _create_buffers(self):
        self._camera_data.intrinsic_matrices = th.zeros((1, 3, 3), device=self._device)
        self._camera_data.intrinsic_matrices[:, 2, 2] = 1.0
        self._camera_data.image_shape = (self.sensor_cfg.pattern_cfg.height, self.sensor_cfg.pattern_cfg.width)
        self._camera_data.output = {}
        self._camera_data.info = {name: None for name in self.sensor_cfg.data_types} 

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        for _key in self._camera_data.info.keys():
            if _key == 'rgb':
                self._camera_data.output[_key] = self._create_rgb()
            elif _key =='distance_to_camera':
                self._camera_data.output[_key] = self._create_depth()

            else:
                raise ValueError(f"CameraCfg data_types are only support [rgb, distance_to_image_plan] ,not a {self.sensor_cfg.data_types}")
    
    def _create_rgb(self):
        self._renderer.update_scene(self._data, camera=self._cam_name)
        self._image = self._renderer.render()
        self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        return th.from_numpy(self._image).to(self._device)
    
    
    def _create_depth(self):
        self._renderer.update_scene(self._data, camera=self._cam_name)
        self._renderer.enable_depth_rendering()
        self._depth_plane = self._renderer.render()
        self._depth_plane = th.from_numpy(self._depth_plane).to(self._device)
        i_indices, j_indices = th.meshgrid(th.arange(self.sensor_cfg.pattern_cfg.height).to(self._device), th.arange(self.sensor_cfg.pattern_cfg.width).to(self._device), indexing='ij')
        x_camera = (i_indices - self._camera_data.intrinsic_matrices[0,0,2]) * \
            self._depth_plane / self._camera_data.intrinsic_matrices[0,0,0]
        y_camera = (j_indices - self._camera_data.intrinsic_matrices[0,1,2]) * \
            self._depth_plane / self._camera_data.intrinsic_matrices[0,1,1]
        self._depth_image = th.sqrt(self._depth_plane**2 + x_camera**2 + y_camera**2)
        self._depth_image = self._depth_plane
        if self.sensor_cfg.depth_clipping_behavior == "max":
            self._depth_image = th.clip(self._depth_image, max=self.sensor_cfg.max_distance)
        elif self.sensor_cfg.depth_clipping_behavior == "zero":
            self._depth_image[self._depth_image > self.sensor_cfg.max_distance] = 0.0
        self._renderer.disable_depth_rendering()
        return self._depth_image.to(self._device)
    
    def _update_intrinsic_matrices(self):
        f_x = (self.sensor_cfg.pattern_cfg.width * self.sensor_cfg.pattern_cfg.focal_length) / self.sensor_cfg.pattern_cfg.horizontal_aperture
        f_y = (self.sensor_cfg.pattern_cfg.height * self.sensor_cfg.pattern_cfg.focal_length) / self.sensor_cfg.pattern_cfg.vertical_aperture
        c_x = self.sensor_cfg.pattern_cfg.horizontal_aperture_offset * f_x + self.sensor_cfg.pattern_cfg.width / 2
        c_y = self.sensor_cfg.pattern_cfg.vertical_aperture_offset * f_y + self.sensor_cfg.pattern_cfg.height / 2
        self._camera_data.intrinsic_matrices[:, 0, 0] = f_x
        self._camera_data.intrinsic_matrices[:, 0, 2] = c_x
        self._camera_data.intrinsic_matrices[:, 1, 1] = f_y
        self._camera_data.intrinsic_matrices[:, 1, 2] = c_y

    def render(self, viewer):
        save_on_headless = True  # change to False if you want to silently skip
        for key, item in self._camera_data.output.items():
            image = item.detach().cpu().numpy() 
            if not self._imshow_initialized:
                plt.ion()  
                self._imshow_handle = plt.imshow(image)
                plt.title("Depth Camera View")
                plt.show(block=False)
                self._imshow_initialized = True
            else:
                self._imshow_handle.set_data(image)
                plt.draw()
                plt.pause(0.001)    
        #     plt.imshow(image)
        #     plt.show()
        # cv2.waitKey(1)

        pass

    @property
    def sensor_data(self) -> CameraData:
        self._update_outdated_buffers()
        return self._camera_data
