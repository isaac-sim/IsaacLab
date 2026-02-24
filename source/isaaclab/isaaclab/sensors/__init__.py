# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package containing various sensor classes implementations.

This subpackage contains the sensor classes that are compatible with Isaac Sim. We include both
USD-based and custom sensors:

* **USD-prim sensors**: Available in Omniverse and require creating a USD prim for them.
  For instance, RTX ray tracing camera and lidar sensors.
* **USD-schema sensors**: Available in Omniverse and require creating a USD schema on an existing prim.
  For instance, contact sensors and frame transformers.
* **Custom sensors**: Implemented in Python and do not require creating any USD prim or schema.
  For instance, warp-based ray-casters.

Due to the above categorization, the prim paths passed to the sensor's configuration class
are interpreted differently based on the sensor type. The following table summarizes the
interpretation of the prim paths for different sensor types:

+---------------------+---------------------------+---------------------------------------------------------------+
| Sensor Type         | Example Prim Path         | Pre-check                                                     |
+=====================+===========================+===============================================================+
| Camera              | /World/robot/base/camera  | Leaf is available, and it will spawn a USD camera             |
+---------------------+---------------------------+---------------------------------------------------------------+
| Contact Sensor      | /World/robot/feet_*       | Leaf is available and checks if the schema exists             |
+---------------------+---------------------------+---------------------------------------------------------------+
| Ray Caster          | /World/robot/base         | Leaf exists and is a physics body (Articulation / Rigid Body) |
+---------------------+---------------------------+---------------------------------------------------------------+
| Frame Transformer   | /World/robot/base         | Leaf exists and is a physics body (Articulation / Rigid Body) |
+---------------------+---------------------------+---------------------------------------------------------------+
| Imu                 | /World/robot/base         | Leaf exists and is a physics body (Rigid Body)                |
+---------------------+---------------------------+---------------------------------------------------------------+

"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["camera", "contact_sensor", "frame_transformer", "imu", "ray_caster"],
    submod_attrs={
        "sensor_base": ["SensorBase"],
        "sensor_base_cfg": ["SensorBaseCfg"],
        "camera": ["Camera", "CameraCfg", "CameraData", "TiledCamera", "TiledCameraCfg", "save_images_to_file"],
        "contact_sensor": [
            "BaseContactSensor",
            "BaseContactSensorData",
            "ContactSensor",
            "ContactSensorCfg",
            "ContactSensorData",
        ],
        "frame_transformer": [
            "BaseFrameTransformer",
            "BaseFrameTransformerData",
            "FrameTransformer",
            "FrameTransformerCfg",
            "FrameTransformerData",
            "OffsetCfg",
        ],
        "imu": ["BaseImu", "BaseImuData", "Imu", "ImuCfg", "ImuData"],
        "ray_caster": [
            "MultiMeshRayCaster",
            "MultiMeshRayCasterCamera",
            "MultiMeshRayCasterCameraCfg",
            "MultiMeshRayCasterCameraData",
            "MultiMeshRayCasterCfg",
            "MultiMeshRayCasterData",
            "RayCaster",
            "RayCasterCamera",
            "RayCasterCameraCfg",
            "RayCasterCfg",
            "RayCasterData",
        ],
    },
)

# Re-export patterns submodule from ray_caster for backward compat
# (from isaaclab.sensors import patterns)
_lazy_getattr = __getattr__


def __getattr__(name):
    if name == "patterns":
        from isaaclab.sensors.ray_caster import patterns

        return patterns
    return _lazy_getattr(name)
