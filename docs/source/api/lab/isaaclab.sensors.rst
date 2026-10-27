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
    Pva
    PvaData
    PvaCfg
    JointWrenchSensor
    JointWrenchSensorData
    JointWrenchSensorCfg

Sensor Base
-----------

.. rubric:: Extending batch updates

Sensors opt into eager batch updates by overriding
:attr:`SensorBase.supports_batch_update`, which defaults to ``False``.
Eager scene updates advance sensors in order, then batch the remaining buffer refreshes for
sensors that opted in. Lazy data access remains per sensor.

For standalone use, :meth:`SensorBase.update_batch` performs the complete eager update for a
sequence of batch-capable sensors. It calls each sensor's ``update()`` once in the supplied order,
then refreshes pending buffers together. Each call advances sensor clocks by ``dt``, so use it
in place of separate ``sensor.update(dt)`` calls. An unsupported sensor raises ``ValueError``
before any input is updated; update unsupported sensors individually.

An implementation can override the static ``_update_buffers_batch_impl(sensors)`` hook to perform
shared work. Sensors inheriting the same hook are grouped together, including instances of
different subclasses. The hook must respect each sensor's outdated-environment mask and fill its
buffers; ``SensorBase`` handles the update timestamps after the hook succeeds. The default hook
calls each sensor's individual buffer implementation.

Camera subclasses that override the individual buffer-update hooks retain individual updates by
default. They must explicitly opt into batching and ensure their batch implementation preserves
the custom behavior.

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

Pose Velocity Acceleration Sensor
---------------------------------

.. autoclass:: Pva
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: PvaData
    :members:
    :inherited-members:
    :exclude-members: __init__

.. autoclass:: PvaCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type

Joint Wrench Sensor
-------------------

.. autoclass:: JointWrenchSensor
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: JointWrenchSensorData
    :members:
    :inherited-members:
    :exclude-members: __init__

.. autoclass:: JointWrenchSensorCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type


Additional Public Classes
-------------------------

The following classes are part of the public :mod:`isaaclab.sensors` API.

.. currentmodule:: isaaclab.sensors

.. autosummary::
   :nosignatures:

   BaseContactSensor
   BaseContactSensorData
   BaseFrameTransformer
   BaseFrameTransformerData
   BaseImu
   BaseImuData
   BaseJointWrenchSensor
   BaseJointWrenchSensorData
   BasePva
   BasePvaData
   ImuData
   MultiMeshRayCasterCameraData
   TiledCamera
   TiledCameraCfg

.. autoclass:: BaseContactSensor
   :show-inheritance:

.. autoclass:: BaseContactSensorData
   :show-inheritance:

.. autoclass:: BaseFrameTransformer
   :show-inheritance:

.. autoclass:: BaseFrameTransformerData
   :show-inheritance:

.. autoclass:: BaseImu
   :show-inheritance:

.. autoclass:: BaseImuData
   :show-inheritance:

.. autoclass:: BaseJointWrenchSensor
   :show-inheritance:

.. autoclass:: BaseJointWrenchSensorData
   :show-inheritance:

.. autoclass:: BasePva
   :show-inheritance:

.. autoclass:: BasePvaData
   :show-inheritance:

.. autoclass:: ImuData
   :show-inheritance:

.. autoclass:: MultiMeshRayCasterCameraData
   :show-inheritance:

.. autoclass:: TiledCamera
   :show-inheritance:

.. autoclass:: TiledCameraCfg
   :show-inheritance:
