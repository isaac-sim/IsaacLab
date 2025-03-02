# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
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

from .camera import *  # noqa: F401, F403
from .contact_sensor import *  # noqa: F401, F403
from .frame_transformer import *  # noqa: F401
from .imu import *  # noqa: F401, F403
from .ray_caster import *  # noqa: F401, F403
from .sensor_base import SensorBase  # noqa: F401
from .sensor_base_cfg import SensorBaseCfg  # noqa: F401
