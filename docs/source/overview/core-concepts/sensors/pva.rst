.. _overview_sensors_pva:

.. currentmodule:: isaaclab

Pose Velocity Acceleration (PVA) Sensor
=======================================

The Pose Velocity Acceleration (PVA) sensor is a ground-truth sensor for reading
the kinematic state of a frame in the simulation. It reports the sensor pose in
the world frame, projected gravity, linear and angular velocities in the sensor
frame, and coordinate accelerations in the sensor frame. Unlike
:class:`~isaaclab.sensors.Imu`, the PVA sensor does not model proper
acceleration from an accelerometer. Use the IMU sensor when the observation
should include accelerometer-like gravity bias behavior.

The sensor can be attached to a rigid body or to a child prim under a rigid-body
ancestor. If the configured prim is not itself rigid, Isaac Lab queries the
closest rigid ancestor and composes the fixed transform to the requested prim
with the configured sensor offset.

Consider a simple environment with an Anymal Quadruped equipped with PVA sensors
on its front feet.

.. literalinclude:: ../../../../../scripts/demos/sensors/pva_sensor.py
  :language: python
  :lines: 43-59

Retrieving values from the sensor follows the same pattern as the other Isaac
Lab sensors. The data fields are exposed as :class:`~isaaclab.utils.warp.ProxyArray`
buffers and can be converted to Torch tensors with the ``torch`` property.

.. code-block:: python

  pva_data = scene["pva_LF"].data
  print("Pose in world frame: ", pva_data.pose_w.torch)
  print("Linear velocity in PVA frame: ", pva_data.lin_vel_b.torch)
  print("Angular velocity in PVA frame: ", pva_data.ang_vel_b.torch)
  print("Linear acceleration in PVA frame: ", pva_data.lin_acc_b.torch)
  print("Angular acceleration in PVA frame: ", pva_data.ang_acc_b.torch)
  print("Projected gravity in PVA frame: ", pva_data.projected_gravity_b.torch)

The complete demo can be run with:

.. code-block:: bash

  ./isaaclab.sh -p scripts/demos/sensors/pva_sensor.py

.. dropdown:: Code for pva_sensor.py
   :icon: code

   .. literalinclude:: ../../../../../scripts/demos/sensors/pva_sensor.py
      :language: python
      :linenos:
