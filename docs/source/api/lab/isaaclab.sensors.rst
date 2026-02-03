isaaclab.sensors
================

.. automodule:: isaaclab.sensors

  .. rubric:: Classes

  .. autosummary::

    SensorBase
    SensorBaseCfg
    Camera
    CameraData
    CameraCfg
    TiledCamera
    TiledCameraCfg
    ContactSensor
    ContactSensorData
    ContactSensorCfg

Sensor Base
-----------

.. autoclass:: SensorBase
    :members:

.. autoclass:: SensorBaseCfg
    :members:
    :exclude-members: __init__, class_type

USD Camera
----------

.. autoclass:: Camera
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: CameraData
    :members:
    :inherited-members:
    :exclude-members: __init__

.. autoclass:: CameraCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type, OffsetCfg

Tile-Rendered USD Camera
------------------------

.. autoclass:: TiledCamera
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: TiledCameraCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type

Contact Sensor
--------------

.. autoclass:: ContactSensor
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: ContactSensorData
    :members:
    :inherited-members:
    :exclude-members: __init__

.. autoclass:: ContactSensorCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type
