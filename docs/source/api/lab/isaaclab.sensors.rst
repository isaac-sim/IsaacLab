isaaclab.sensors
================

.. automodule:: isaaclab.sensors

  .. rubric:: Submodules

  .. autosummary::

    patterns

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
    FrameTransformer
    FrameTransformerData
    FrameTransformerCfg
    RayCaster
    RayCasterData
    RayCasterCfg
    RayCasterCamera
    RayCasterCameraCfg
    MultiMeshRayCaster
    MultiMeshRayCasterData
    MultiMeshRayCasterCfg
    MultiMeshRayCasterCamera
    MultiMeshRayCasterCameraCfg
    Imu
    ImuCfg

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

Frame Transformer
-----------------

.. autoclass:: FrameTransformer
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: FrameTransformerData
    :members:
    :inherited-members:
    :exclude-members: __init__

.. autoclass:: FrameTransformerCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type

.. autoclass:: OffsetCfg
    :members:
    :inherited-members:
    :exclude-members: __init__

Ray-Cast Sensor
---------------

.. autoclass:: RayCaster
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: RayCasterData
    :members:
    :inherited-members:
    :exclude-members: __init__

.. autoclass:: RayCasterCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type

Ray-Cast Camera
---------------

.. autoclass:: RayCasterCamera
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: RayCasterCameraCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type, OffsetCfg

Multi-Mesh Ray-Cast Sensor
--------------------------

.. autoclass:: MultiMeshRayCaster
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: MultiMeshRayCasterData
    :members:
    :inherited-members:
    :exclude-members: __init__

.. autoclass:: MultiMeshRayCasterCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type, OffsetCfg

Multi-Mesh Ray-Cast Camera
--------------------------

.. autoclass:: MultiMeshRayCasterCamera
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: MultiMeshRayCasterCameraCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type, OffsetCfg, RaycastTargetCfg

Inertia Measurement Unit
------------------------

.. autoclass:: Imu
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: ImuCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type
