.. _overview_sensors_imu:

Inertial Measurement Unit (IMU)
===================================

.. figure:: ../../_static/overview/overview_sensors_imu_diagram.jpg
    :align: center
    :figwidth: 100%
    :alt: A diagram outlining the basic force relationships for the IMU sensor

Inertial Measurement Units (IMUs) are a type of sensor for measuring the acceleration of an object.  These sensors are traditionally designed report linear accelerations and angular velocities, and function on similar principles to that of a digital scale: They report accelerations derived from **net force acting on the sensor**.

A naive implementation of an IMU would report a negative acceleration due to gravity while the sensor is at rest in some local gravitational field. This is not generally needed for most practical applications, and so most real IMU sensors often include a **gravity bias** and assume that the device is operating on the surface of the Earth.  The IMU we provide in Isaac Lab includes a similar bias term, which defaults to +g.  This means that if you add an IMU to your simulation, and do not change this bias term, you will detect an acceleration of :math:`+ 9.81 m/s^{2}` anti-parallel to gravity acceleration.

Consider a simple environment with an Anymal Quadruped equipped with an IMU on each of its two front feet.

.. literalinclude:: ../../../../scripts/demos/sensors/imu_sensor.py
  :language: python
  :lines: 39-63

Here we have explicitly removed the bias from one of the sensors, and we can see how this affects the reported values by visualizing the sensor when we run the sample script

.. figure:: ../../_static/overview/overview_sensors_imu_visualizer.jpg
    :align: center
    :figwidth: 100%
    :alt: IMU visualized

Notice that the right front foot explicitly has a bias of (0,0,0).  In the visualization, you should see that the arrow indicating the acceleration from the right IMU rapidly changes over time, while the arrow visualizing the left IMU points constantly along the vertical axis.

Retrieving values form the sensor is done in the usual way

.. code-block:: python

  def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    .
    .
    .
    # Simulate physics
    while simulation_app.is_running():
      .
      .
      .
      # print information from the sensors
      print("-------------------------------")
      print(scene["imu_LF"])
      print("Received linear velocity: ", scene["imu_LF"].data.lin_vel_b)
      print("Received angular velocity: ", scene["imu_LF"].data.ang_vel_b)
      print("Received linear acceleration: ", scene["imu_LF"].data.lin_acc_b)
      print("Received angular acceleration: ", scene["imu_LF"].data.ang_acc_b)
      print("-------------------------------")
      print(scene["imu_RF"])
      print("Received linear velocity: ", scene["imu_RF"].data.lin_vel_b)
      print("Received angular velocity: ", scene["imu_RF"].data.ang_vel_b)
      print("Received linear acceleration: ", scene["imu_RF"].data.lin_acc_b)
      print("Received angular acceleration: ", scene["imu_RF"].data.ang_acc_b)

The oscillations in the values reported by the sensor are a direct result of of how the sensor calculates the acceleration, which is through a finite difference approximation between adjacent ground truth velocity values as reported by the sim.  We can see this in the reported result (pay attention to the **linear acceleration**) because the acceleration from the right foot is small, but explicitly zero.

.. code-block:: bash

  Imu sensor @ '/World/envs/env_.*/Robot/LF_FOOT':
          view type         : <class 'omni.physics.tensors.impl.api.RigidBodyView'>
          update period (s) : 0.0
          number of sensors : 1

  Received linear velocity:  tensor([[ 0.0203, -0.0054,  0.0380]], device='cuda:0')
  Received angular velocity:  tensor([[-0.0104, -0.1189,  0.0080]], device='cuda:0')
  Received linear acceleration:  tensor([[ 4.8344, -0.0205,  8.5305]], device='cuda:0')
  Received angular acceleration:  tensor([[-0.0389, -0.0262, -0.0045]], device='cuda:0')
  -------------------------------
  Imu sensor @ '/World/envs/env_.*/Robot/RF_FOOT':
          view type         : <class 'omni.physics.tensors.impl.api.RigidBodyView'>
          update period (s) : 0.0
          number of sensors : 1

  Received linear velocity:  tensor([[0.0244, 0.0077, 0.0431]], device='cuda:0')
  Received angular velocity:  tensor([[ 0.0122, -0.1360, -0.0042]], device='cuda:0')
  Received linear acceleration:  tensor([[-0.0018,  0.0010, -0.0032]], device='cuda:0')
  Received angular acceleration:  tensor([[-0.0373, -0.0050, -0.0053]], device='cuda:0')
  -------------------------------

.. dropdown:: Code for imu_sensor.py
   :icon: code

   .. literalinclude:: ../../../../scripts/demos/sensors/imu_sensor.py
      :language: python
      :linenos:
