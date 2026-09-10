.. _concepts_sensors_pva:
.. _overview_sensors_pva:

.. currentmodule:: isaaclab

Pose Velocity Acceleration (PVA) Sensor
=======================================

A :class:`~sensors.Pva` reads the ground-truth kinematic state of a frame. Use it for privileged
observations, state estimation, or control inputs that require more than an inertial sensor measures.
Use an :doc:`imu` when the observation should follow accelerometer and gyroscope conventions.

The sensor can attach directly to a rigid body or to a fixed child prim beneath a rigid-body
ancestor. In the latter case, Isaac Lab composes the child's fixed transform with the configured
sensor offset.

Measurement contract
--------------------

For ``E`` environments, every vector has shape ``(E, 3)``. The pose has shape ``(E, 7)`` in
``(x, y, z, qx, qy, qz, qw)`` order.

.. list-table::
   :header-rows: 1
   :widths: 28 27 45

   * - Buffer
     - Frame and units
     - Meaning
   * - ``pose_w``
     - World; [m, unitless]
     - Sensor position and orientation
   * - ``projected_gravity_b``
     - Sensor; unitless
     - Unit gravity direction projected into the sensor frame
   * - ``lin_vel_b``
     - Sensor; [m/s]
     - Linear velocity relative to the world
   * - ``ang_vel_b``
     - Sensor; [rad/s]
     - Angular velocity relative to the world
   * - ``lin_acc_b``
     - Sensor; [m/s²]
     - Coordinate linear acceleration; zero at rest and :math:`-g` in free fall
   * - ``ang_acc_b``
     - Sensor; [rad/s²]
     - Angular acceleration relative to the world

Configure and read the sensor
-----------------------------

.. code-block:: python

   from isaaclab.sensors import PvaCfg

   base_state = PvaCfg(
       prim_path="{ENV_REGEX_NS}/Robot/base",
       update_period=0.0,
       offset=PvaCfg.OffsetCfg(pos=(0.0, 0.0, 0.05)),
   )

The data fields are :class:`~isaaclab.utils.warp.ProxyArray` buffers. Convert them to Torch views only
where Torch operations are required:

.. code-block:: python

   pva_data = scene["base_state"].data
   pose_w = pva_data.pose_w.torch
   linear_velocity = pva_data.lin_vel_b.torch
   coordinate_acceleration = pva_data.lin_acc_b.torch

Like the IMU, acceleration uses state from consecutive simulation updates. Reset the scene and sensor
state together at episode boundaries.

A complete runnable example is available in ``scripts/demos/sensors/pva_sensor.py``:

.. code-block:: bash

   uv run --extra isaacsim python scripts/demos/sensors/pva_sensor.py
