.. _concepts_sensors:
.. _overview_sensors:

Sensors
========

Sensors provide batched measurements of simulation state and camera images.

Choose a sensor
---------------

.. list-table::
   :header-rows: 1
   :widths: 28 47 25

   * - Sensor
     - Measurement
     - Typical use
   * - :doc:`camera`
     - Renderer-produced color, depth, normals, motion, and segmentation images
     - Vision observations and synthetic data
   * - :doc:`ray_caster`
     - First collision locations for configurable ray patterns
     - Terrain sensing, lidar-style queries, and ray-cast cameras
   * - :doc:`contact_sensor`
     - Net and filtered contact forces, contact points, friction, and contact timing
     - Locomotion, grasping, and collision events
   * - :doc:`frame_transformer`
     - Relative and world poses for configured frames
     - End-effector, foot, and object tracking
   * - :doc:`imu`
     - Angular velocity and proper linear acceleration
     - Inertial observations
   * - :doc:`pva`
     - Ground-truth pose, velocity, coordinate acceleration, and projected gravity
     - State estimation and privileged observations
   * - :doc:`joint_wrench_sensor`
     - Incoming joint reaction force and torque
     - Force/torque sensing and contact-rich control

Sampling and data access
------------------------

All sensors inherit from :class:`~isaaclab.sensors.SensorBase` and share these settings
and methods:

* ``prim_path`` selects the prim or prims measured in every cloned environment.
* ``update_period`` sets the sampling period in simulated seconds. A value of ``0.0`` samples every
  simulation step.
* :meth:`~isaaclab.sensors.SensorBase.update` advances the sensor clock. Sensor data is evaluated
  lazily when the :attr:`~isaaclab.sensors.SensorBase.data` property is read, unless recomputation or
  debug visualization is requested.
* :meth:`~isaaclab.sensors.SensorBase.reset` clears per-environment timestamps and internal state.

Set ``scene.cfg.lazy_sensor_update=False`` to refresh data during ``scene.update(dt)``.
Sensors keep their own sampling periods and reset state. Sensors that support
batching refresh together; cameras :ref:`batch captures by renderer <renderer-camera-batching>`.

Sensor data is exposed through :class:`~isaaclab.utils.warp.ProxyArray` buffers, including camera
outputs. Use the ``torch`` property for a cached zero-copy Torch view or ``warp`` for the underlying
Warp array.

.. toctree::
   :maxdepth: 1
   :hidden:

   camera
   ray_caster
   contact_sensor
   frame_transformer
   imu
   pva
   joint_wrench_sensor

.. Maintenance notes

   These pages are the source of truth for the ``isaaclab-using-sensors-actuators`` agent skill
   (`skills/user/use-sensors-actuators/ <../../../../skills/user/use-sensors-actuators/SKILL.md>`__).
   When sensor behavior changes, keep the API documentation, maintained demos, these concept pages,
   and the skill synchronized.
