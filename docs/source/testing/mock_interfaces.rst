.. _testing_mock_interfaces:

Mock Interfaces for Unit Testing
================================

Isaac Lab provides mock implementations of sensor, asset, and PhysX view classes
for unit testing without requiring Isaac Sim or GPU simulation.

Overview
--------

Two levels of mock interfaces are available:

1. **High-level mocks** (``isaaclab.test.mock_interfaces``) - Mock sensors and assets
2. **Low-level PhysX mocks** (``isaaclab_physx.test.mock_interfaces``) - Mock PhysX TensorAPI views

High-Level Mock Interfaces
--------------------------

Located in ``isaaclab.test.mock_interfaces``, these mock the public API of sensor
and asset base classes.

Available Mocks
~~~~~~~~~~~~~~~

**Sensors:**

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Mock Class
     - Real Class
     - Description
   * - ``MockImu``
     - ``BaseImu``
     - IMU sensor (accelerometer + gyroscope)
   * - ``MockContactSensor``
     - ``BaseContactSensor``
     - Contact force sensor
   * - ``MockFrameTransformer``
     - ``BaseFrameTransformer``
     - Frame tracking sensor

**Assets:**

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Mock Class
     - Real Class
     - Description
   * - ``MockArticulation``
     - ``BaseArticulation``
     - Articulated robot (joints + bodies)
   * - ``MockRigidObject``
     - ``BaseRigidObject``
     - Single rigid body
   * - ``MockRigidObjectCollection``
     - ``BaseRigidObjectCollection``
     - Collection of rigid bodies

Quick Start
~~~~~~~~~~~

.. code-block:: python

   import torch
   from isaaclab.test.mock_interfaces import MockArticulation, MockContactSensor

   # Create a mock quadruped robot
   robot = MockArticulation(
       num_instances=4,
       num_joints=12,
       num_bodies=13,
       joint_names=[f"joint_{i}" for i in range(12)],
       device="cpu"
   )

   # Set joint states
   robot.data.set_mock_data(
       joint_pos=torch.zeros(4, 12),
       joint_vel=torch.randn(4, 12) * 0.1
   )

   # Access data like the real asset
   positions = robot.data.joint_pos  # Shape: (4, 12)

   # Create a mock contact sensor
   sensor = MockContactSensor(num_instances=4, num_bodies=4, device="cpu")
   sensor.data.set_net_forces_w(torch.randn(4, 4, 3))

Factory Functions
~~~~~~~~~~~~~~~~~

Pre-configured factories for common use cases:

.. code-block:: python

   from isaaclab.test.mock_interfaces import (
       create_mock_quadruped,
       create_mock_humanoid,
       create_mock_foot_contact_sensor,
   )

   # Pre-configured quadruped (12 joints, 13 bodies)
   robot = create_mock_quadruped(num_instances=4)

   # Pre-configured humanoid (21 joints, 22 bodies)
   humanoid = create_mock_humanoid(num_instances=2)

   # Foot contact sensor for quadruped
   foot_contact = create_mock_foot_contact_sensor(num_instances=4, num_feet=4)

Patching Utilities
~~~~~~~~~~~~~~~~~~

Use context managers or decorators to inject mocks into tests:

**Context Managers:**

.. code-block:: python

   from isaaclab.test.mock_interfaces.utils import patch_articulation, patch_sensor

   with patch_articulation("my_module.Articulation", num_joints=12) as MockRobot:
       robot = my_function_that_creates_robot()
       robot.data.set_joint_pos(torch.zeros(1, 12))

   with patch_sensor("my_module.ContactSensor", "contact", num_bodies=4):
       sensor = my_function_that_creates_sensor()

**Decorators:**

.. code-block:: python

   from isaaclab.test.mock_interfaces.utils import mock_articulation, mock_sensor

   @mock_articulation(num_joints=12, num_bodies=13)
   def test_observation_function(mock_robot):
       mock_robot.data.set_joint_pos(torch.zeros(1, 12))
       obs = compute_observation(mock_robot)
       assert obs.shape == (1, 24)

   @mock_sensor("contact", num_instances=4, num_bodies=4)
   def test_contact_reward(mock_contact):
       mock_contact.data.set_net_forces_w(torch.randn(4, 4, 3))
       reward = compute_contact_reward(mock_contact)

Low-Level PhysX Mock Interfaces
-------------------------------

Located in ``isaaclab_physx.test.mock_interfaces``, these mock the PhysX TensorAPI
views used internally by assets and sensors.

Available Mocks
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Mock Class
     - PhysX Class
     - Used By
   * - ``MockRigidBodyView``
     - ``physx.RigidBodyView``
     - ``RigidObject``
   * - ``MockArticulationView``
     - ``physx.ArticulationView``
     - ``Articulation``
   * - ``MockRigidContactView``
     - ``physx.RigidContactView``
     - ``ContactSensor``

Quick Start
~~~~~~~~~~~

.. code-block:: python

   from isaaclab_physx.test.mock_interfaces import (
       MockRigidBodyView,
       MockArticulationView,
       MockRigidContactView,
   )

   # Create a mock rigid body view
   view = MockRigidBodyView(count=4, device="cpu")
   transforms = view.get_transforms()  # Shape: (4, 7)

   # Set mock data
   view.set_mock_transforms(torch.randn(4, 7))

   # Create a mock articulation view
   art_view = MockArticulationView(
       count=4,
       num_dofs=12,
       num_links=13,
       device="cpu"
   )
   positions = art_view.get_dof_positions()  # Shape: (4, 12)

Factory Functions
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from isaaclab_physx.test.mock_interfaces import (
       create_mock_rigid_body_view,
       create_mock_articulation_view,
       create_mock_quadruped_view,
       create_mock_rigid_contact_view,
   )

   # Pre-configured quadruped view (12 DOFs, 13 links)
   view = create_mock_quadruped_view(count=4)

   # Pre-configured humanoid view (21 DOFs, 22 links)
   humanoid = create_mock_humanoid_view(count=2)

Patching Utilities
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from isaaclab_physx.test.mock_interfaces.utils import (
       patch_rigid_body_view,
       patch_articulation_view,
       mock_articulation_view,
   )

   # Context manager
   with patch_rigid_body_view("my_module.physx.RigidBodyView", count=4):
       view = create_rigid_body()

   # Decorator
   @mock_articulation_view(count=4, num_dofs=12, num_links=13)
   def test_my_function(mock_view):
       positions = mock_view.get_dof_positions()
       assert positions.shape == (4, 12)

Data Shapes Reference
---------------------

High-Level Mocks
~~~~~~~~~~~~~~~~

**IMU Data:**

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Property
     - Shape
     - Description
   * - ``pos_w``
     - ``(N, 3)``
     - Position in world frame
   * - ``quat_w``
     - ``(N, 4)``
     - Orientation (w,x,y,z) in world frame
   * - ``lin_vel_b``
     - ``(N, 3)``
     - Linear velocity in body frame
   * - ``ang_vel_b``
     - ``(N, 3)``
     - Angular velocity in body frame
   * - ``projected_gravity_b``
     - ``(N, 3)``
     - Gravity in body frame

**Contact Sensor Data:**

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Property
     - Shape
     - Description
   * - ``net_forces_w``
     - ``(N, B, 3)``
     - Net contact forces
   * - ``force_matrix_w``
     - ``(N, B, M, 3)``
     - Filtered forces
   * - ``current_contact_time``
     - ``(N, B)``
     - Time in contact
   * - ``current_air_time``
     - ``(N, B)``
     - Time in air

Where N = instances, B = bodies, M = filter bodies

**Articulation Data:**

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Property
     - Shape
     - Description
   * - ``joint_pos``
     - ``(N, J)``
     - Joint positions
   * - ``joint_vel``
     - ``(N, J)``
     - Joint velocities
   * - ``root_link_pose_w``
     - ``(N, 7)``
     - Root pose in world
   * - ``body_link_pose_w``
     - ``(N, B, 7)``
     - Body poses in world

Where N = instances, J = joints, B = bodies

PhysX View Mocks
~~~~~~~~~~~~~~~~

**RigidBodyView:**

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Method
     - Shape
     - Description
   * - ``get_transforms()``
     - ``(N, 7)``
     - [pos(3), quat_xyzw(4)]
   * - ``get_velocities()``
     - ``(N, 6)``
     - [lin_vel(3), ang_vel(3)]
   * - ``get_masses()``
     - ``(N, 1, 1)``
     - mass per body
   * - ``get_inertias()``
     - ``(N, 1, 3, 3)``
     - 3x3 inertia matrix

**ArticulationView:**

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Method
     - Shape
     - Description
   * - ``get_root_transforms()``
     - ``(N, 7)``
     - Root pose
   * - ``get_link_transforms()``
     - ``(N, L, 7)``
     - Link poses
   * - ``get_dof_positions()``
     - ``(N, J)``
     - Joint positions
   * - ``get_dof_limits()``
     - ``(N, J, 2)``
     - [lower, upper]

**RigidContactView:**

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Method
     - Shape
     - Description
   * - ``get_net_contact_forces(dt)``
     - ``(N*B, 3)``
     - Flat net forces
   * - ``get_contact_force_matrix(dt)``
     - ``(N*B, F, 3)``
     - Per-filter forces

Design Patterns
---------------

All mock interfaces follow these patterns:

1. **Lazy Initialization** - Tensors created on first access with correct shapes
2. **Device Transfer** - All setters call ``.to(device)``
3. **Identity Quaternion Defaults** - Quaternions default to identity
4. **Clone on Getters** - Return ``.clone()`` to prevent mutation
5. **No-op Actions** - Simulation operations (reset, update, write_to_sim) do nothing
6. **Mock Setters** - Direct ``set_mock_*`` methods for test setup

Example: Testing an Observation Function
----------------------------------------

.. code-block:: python

   import pytest
   import torch
   from isaaclab.test.mock_interfaces import MockArticulation

   @pytest.fixture
   def robot():
       return MockArticulation(
           num_instances=4,
           num_joints=12,
           num_bodies=13,
           device="cpu"
       )

   def test_joint_observation(robot):
       # Setup
       joint_pos = torch.randn(4, 12)
       robot.data.set_joint_pos(joint_pos)

       # Test your observation function
       obs = compute_joint_observation(robot)

       # Verify
       assert obs.shape == (4, 24)  # pos + vel
       assert torch.allclose(obs[:, :12], joint_pos)

   def test_body_observation(robot):
       # Setup
       body_poses = torch.randn(4, 13, 7)
       robot.data.set_body_link_pose_w(body_poses)

       # Test
       obs = compute_body_observation(robot)
       assert obs.shape == (4, 13, 7)

Example: Testing with PhysX View Mocks
--------------------------------------

.. code-block:: python

   import pytest
   import torch
   from isaaclab_physx.test.mock_interfaces import create_mock_quadruped_view
   from isaaclab_physx.test.mock_interfaces.utils import mock_articulation_view

   def test_quadruped_joint_limits():
       """Test quadruped joint limit handling."""
       view = create_mock_quadruped_view(count=4)

       # Set custom joint limits
       limits = torch.zeros(4, 12, 2)
       limits[:, :, 0] = -1.5  # lower limit
       limits[:, :, 1] = 1.5   # upper limit
       view.set_mock_dof_limits(limits)

       # Verify limits
       result = view.get_dof_limits()
       assert torch.allclose(result[:, :, 0], torch.full((4, 12), -1.5))
       assert torch.allclose(result[:, :, 1], torch.full((4, 12), 1.5))

   @mock_articulation_view(count=4, num_dofs=12, num_links=13)
   def test_articulation_with_decorator(mock_view):
       """Test using the decorator pattern."""
       # Set some initial positions
       positions = torch.randn(4, 12)
       mock_view.set_mock_dof_positions(positions)

       # Verify
       result = mock_view.get_dof_positions()
       assert torch.allclose(result, positions)
