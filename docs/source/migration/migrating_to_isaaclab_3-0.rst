.. _migrating-to-isaaclab-3-0:

Migrating to Isaac Lab 3.0
==========================

.. currentmodule:: isaaclab

Isaac Lab 3.0 introduces a multi-backend architecture that separates simulation backend-specific code
from the core Isaac Lab API. This allows for future support of different physics backends while
maintaining a consistent user-facing API.

This guide covers the main breaking changes and deprecations you need to address when migrating
from Isaac Lab 2.x to Isaac Lab 3.0.


Visualizer CLI and Headless Behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Isaac Lab 3.0, the ``--headless`` argument is deprecated. Instead, use ``--visualizer`` / ``--viz``
to determine whether viewer apps are launched with an Isaac Lab command.

Visualizers are lightweight viewer apps for monitoring, debugging, and recording workflows
(see :doc:`/source/features/visualization`).

The details below describe how CLI visualizer arguments resolve together with
``SimulationCfg.visualizer_cfgs``.

- ``--viz`` accepts **comma-separated** values (for example ``--viz kit,newton``).
- If omitted, visualizers are resolved from ``SimulationCfg.visualizer_cfgs``.
- ``--viz none`` explicitly disables all visualizers, including config-defined ones.
- ``--headless`` is deprecated (still supported) and overrides ``--viz`` by forcing headless mode.

For the full behavior of visualizer resolution, with the visualizer CLI arg, visualizer configs,
and ``--headless``, see :ref:`visualization-common-modes`.


Multi-Backend Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~

Isaac Lab 3.0 introduces a **factory-based multi-backend architecture** that allows asset classes
to be backed by different physics engines — currently **PhysX** and **Newton**.

When you instantiate an asset class from the ``isaaclab`` package (e.g., ``Articulation``,
``RigidObject``), a factory automatically resolves and loads the correct backend implementation:

.. code-block:: python

   from isaaclab.assets import Articulation, ArticulationCfg

   # The factory pattern creates the appropriate backend implementation.
   # No import changes are needed — the same isaaclab imports work regardless of backend.
   robot = Articulation(cfg=ArticulationCfg(...))

The factory works by convention: for a class defined in ``isaaclab.assets.articulation``, it
imports the matching class from ``isaaclab_{backend}.assets.articulation``. This means the
``isaaclab_physx`` and ``isaaclab_newton`` packages mirror the ``isaaclab`` module structure.

.. note::

   The backend is currently set to ``"physx"`` by default. Newton backend support is being
   actively developed. When backend selection is fully configurable, you will be able to
   switch backends without changing any asset import paths.

For a comprehensive overview of the factory pattern, backend selection, and how to add a new
backend, see :doc:`/source/overview/core-concepts/multi_backend_architecture`.

New ``isaaclab_physx`` and ``isaaclab_newton`` Extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two new backend extensions have been introduced:

- **``isaaclab_physx``** — PhysX-specific implementations of all asset and sensor classes.
- **``isaaclab_newton``** — Newton-specific implementations of asset classes (Articulation and
  RigidObject).

The following classes have been moved to ``isaaclab_physx``:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Isaac Lab 2.x
     - Isaac Lab 3.0
   * - ``from isaaclab.assets import DeformableObject``
     - ``from isaaclab_physx.assets import DeformableObject``
   * - ``from isaaclab.assets import DeformableObjectCfg``
     - ``from isaaclab_physx.assets import DeformableObjectCfg``
   * - ``from isaaclab.assets import DeformableObjectData``
     - ``from isaaclab_physx.assets import DeformableObjectData``
   * - ``from isaaclab.assets import SurfaceGripper``
     - ``from isaaclab_physx.assets import SurfaceGripper``
   * - ``from isaaclab.assets import SurfaceGripperCfg``
     - ``from isaaclab_physx.assets import SurfaceGripperCfg``

.. note::

   The ``isaaclab_physx`` extension is installed automatically with Isaac Lab. No additional
   installation steps are required.


Renaming of ``XformPrimView`` to ``FrameView``
-----------------------------------------------

Isaac Lab's ``XformPrimView`` and related classes have been renamed to ``FrameView`` to
better reflect their purpose and avoid confusion with Isaac Sim's ``XFormPrim`` class
hierarchy. The old ``XformPrimView`` name is kept as a deprecated alias.

The rename applies across all backends:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Isaac Lab 2.x
     - Isaac Lab 3.0
   * - ``BaseXformPrimView``
     - :class:`~isaaclab.sim.views.BaseFrameView`
   * - ``UsdXformPrimView``
     - :class:`~isaaclab.sim.views.UsdFrameView`
   * - ``XformPrimView``
     - :class:`~isaaclab.sim.views.FrameView`
   * - ``FabricXformPrimView``
     - :class:`~isaaclab_physx.sim.views.FabricFrameView`
   * - ``NewtonSiteXformPrimView``
     - :class:`~isaaclab_newton.sim.views.NewtonSiteFrameView`

For most users, the only change needed is updating imports:

.. code-block:: python

   # Before
   from isaaclab.sim.views import XformPrimView

   # After
   from isaaclab.sim.views import FrameView

The :class:`~isaaclab.sim.views.FrameView` factory automatically dispatches to the correct
backend (:class:`~isaaclab_physx.sim.views.FabricFrameView` for PhysX,
:class:`~isaaclab_newton.sim.views.NewtonSiteFrameView` for Newton) based on the active
physics backend. The deprecated ``XformPrimView`` alias continues to work but will be
removed in a future release.


Unchanged Imports
-----------------

The following asset classes remain in the ``isaaclab`` package and can still be imported as before:

- :class:`~isaaclab.assets.Articulation`, :class:`~isaaclab.assets.ArticulationCfg`, :class:`~isaaclab.assets.ArticulationData`
- :class:`~isaaclab.assets.RigidObject`, :class:`~isaaclab.assets.RigidObjectCfg`, :class:`~isaaclab.assets.RigidObjectData`
- :class:`~isaaclab.assets.RigidObjectCollection`, :class:`~isaaclab.assets.RigidObjectCollectionCfg`, :class:`~isaaclab.assets.RigidObjectCollectionData`

These classes now inherit from new abstract base classes but maintain full backward compatibility.

The following sensor classes also remain in the ``isaaclab`` package with unchanged imports:

- :class:`~isaaclab.sensors.ContactSensor`, :class:`~isaaclab.sensors.ContactSensorCfg`, :class:`~isaaclab.sensors.ContactSensorData`
- :class:`~isaaclab.sensors.Imu`, :class:`~isaaclab.sensors.ImuCfg`, :class:`~isaaclab.sensors.ImuData`
- :class:`~isaaclab.sensors.Pva`, :class:`~isaaclab.sensors.PvaCfg`, :class:`~isaaclab.sensors.PvaData`
- :class:`~isaaclab.sensors.FrameTransformer`, :class:`~isaaclab.sensors.FrameTransformerCfg`, :class:`~isaaclab.sensors.FrameTransformerData`
- :class:`~isaaclab.sensors.JointWrenchSensor`, :class:`~isaaclab.sensors.JointWrenchSensorCfg`,
  :class:`~isaaclab.sensors.JointWrenchSensorData`

These sensor classes now use factory patterns that automatically instantiate the appropriate backend
implementation (PhysX by default), maintaining full backward compatibility.

.. note::

   The ``Imu`` sensor in Isaac Lab 3.0 is **not** the same as the ``Imu`` sensor in 2.x.
   The old ``Imu`` (full state sensor) has been renamed to :class:`~isaaclab.sensors.Pva`.
   The new :class:`~isaaclab.sensors.Imu` is a lightweight sensor that only provides angular velocity
   and linear acceleration. See :ref:`imu-to-pva-migration` below for details.

If you need to import the PhysX sensor implementations directly (e.g., for type hints or subclassing),
you can import from ``isaaclab_physx.sensors``:

.. code-block:: python

   # Direct PhysX implementation imports
   from isaaclab_physx.sensors import ContactSensor, ContactSensorData
   from isaaclab_physx.sensors import Imu, ImuData
   from isaaclab_physx.sensors import Pva, PvaData
   from isaaclab_physx.sensors import FrameTransformer, FrameTransformerData
   from isaaclab_physx.sensors import JointWrenchSensor, JointWrenchSensorData


New ``isaaclab_newton`` Extension
---------------------------------

A new extension ``isaaclab_newton`` provides Newton physics backend implementations for:

- :class:`~isaaclab_newton.assets.Articulation` and :class:`~isaaclab_newton.assets.ArticulationData`
- :class:`~isaaclab_newton.assets.RigidObject` and :class:`~isaaclab_newton.assets.RigidObjectData`
- :class:`~isaaclab_newton.sensors.JointWrenchSensor` and
  :class:`~isaaclab_newton.sensors.JointWrenchSensorData`

These classes implement the same base interfaces as their PhysX counterparts
(:class:`~isaaclab.assets.BaseArticulation`, :class:`~isaaclab.assets.BaseRigidObject`),
ensuring a consistent API across backends. They use the same warp-based data conventions
(``wp.array`` with structured types, ``_index`` / ``_mask`` write methods).

.. note::

   The ``isaaclab_newton`` extension requires the ``newton`` package and its dependencies
   (``mujoco``, ``mujoco-warp``). These are installed automatically when installing the
   ``isaaclab_newton`` package.

If you need to import Newton implementations directly (e.g., for type hints or subclassing):

.. code-block:: python

   from isaaclab_newton.assets import Articulation as NewtonArticulation
   from isaaclab_newton.assets import RigidObject as NewtonRigidObject


Deformable Object API Changes
------------------------------

Isaac Lab 3.0 updates the deformable body API to align with the current Omni Physics 110.0
release. The old soft body API has been deprecated and replaced by two distinct deformable
types: **volume deformables** (3D FEM tetrahedral meshes) and **surface deformables** (2D
triangle cloth meshes). The deformable type is determined by the physics material assigned:

- :class:`~isaaclab_physx.sim.DeformableBodyMaterialCfg` for volume deformables.
- :class:`~isaaclab_physx.sim.SurfaceDeformableBodyMaterialCfg` for surface deformables.

All deformable-related classes have moved from ``isaaclab`` to ``isaaclab_physx``, as shown
in the import table above. Several properties on
:class:`~isaaclab_physx.sim.DeformableBodyPropertiesCfg` have been removed or added to match
the new Omni Physics schema.

For a comprehensive guide covering the full deformable API migration — including removed and
added properties, material changes, code examples for both volume and surface deformables, and
current limitations — see :ref:`migrating-deformables`.


.. _imu-to-pva-migration:

IMU Sensor Renamed to PVA; New Lightweight IMU Sensor
-----------------------------------------------------

The old ``Imu`` sensor has been renamed to **PVA** (Pose Velocity Acceleration) because it provided
full pose, velocity, and acceleration data — far more than a real inertial measurement unit measures.
A new lightweight **IMU** sensor has been introduced that only provides the two physical quantities
a real IMU measures: angular velocity (gyroscope) and linear acceleration (accelerometer).

If you were using the old ``Imu`` sensor, you need to decide which new sensor to use:

- Use :class:`~isaaclab.sensors.Pva` / :class:`~isaaclab.sensors.PvaCfg` if you need full state
  data (pose, linear velocity, angular velocity, linear and angular acceleration, projected gravity).
- Use :class:`~isaaclab.sensors.Imu` / :class:`~isaaclab.sensors.ImuCfg` if you only need angular
  velocity and linear acceleration (as a real IMU provides).

**Import changes:**

.. code-block:: python

   # Before (Isaac Lab 2.x) — the old IMU provided full state
   from isaaclab.sensors import Imu, ImuCfg, ImuData

   # After (Isaac Lab 3.x) — use PVA for the same full-state sensor
   from isaaclab.sensors import Pva, PvaCfg, PvaData

   # Or use the new lightweight IMU for angular velocity + linear acceleration only
   from isaaclab.sensors import Imu, ImuCfg, ImuData

**Configuration changes:**

The ``gravity_bias`` configuration parameter has been removed from both sensors:

- **PVA** reports raw kinematic acceleration (no gravity contribution), as the acceleration
  is derived from finite differencing of velocities which do not include gravity.
- **IMU** unconditionally includes gravity in its accelerometer readings, matching the behavior
  of a real accelerometer. The gravity vector is automatically queried from the simulation.

.. code-block:: python

   # Before (Isaac Lab 2.x)
   imu_cfg = ImuCfg(
       prim_path="{ENV_REGEX_NS}/Robot/base",
       gravity_bias=(0.0, 0.0, 9.81),  # had to be configured manually
   )

   # After (Isaac Lab 3.x) — PVA (no gravity in acceleration)
   pva_cfg = PvaCfg(prim_path="{ENV_REGEX_NS}/Robot/base")

   # After (Isaac Lab 3.x) — IMU (gravity always included automatically)
   imu_cfg = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base")

**Observation function changes:**

.. code-block:: python

   # Before (Isaac Lab 2.x)
   from isaaclab.envs.mdp import imu_orientation, imu_projected_gravity

   # After (Isaac Lab 3.x)
   from isaaclab.envs.mdp import pva_orientation, pva_projected_gravity

**Data property changes:**

The new ``ImuData`` only provides ``ang_vel_b`` and ``lin_acc_b``. If you were accessing other
properties (``pos_w``, ``quat_w``, ``lin_vel_b``, ``ang_acc_b``, ``projected_gravity_b``), switch
to :class:`~isaaclab.sensors.PvaData` which provides all of them.


Sensor Pose Properties Deprecation
----------------------------------

The ``pose_w``, ``pos_w``, and ``quat_w`` properties on :class:`~isaaclab.sensors.ContactSensorData`
are deprecated and will be removed in a future release.

If you need to track sensor poses in world frame, please use a dedicated sensor such as
:class:`~isaaclab.sensors.FrameTransformer` instead.

**Before (deprecated):**

.. code-block:: python

   # Using pose properties directly on sensor data
   sensor_pos = contact_sensor.data.pos_w
   sensor_quat = contact_sensor.data.quat_w

**After (recommended):**

.. code-block:: python

   # Use FrameTransformer to track sensor pose
   frame_transformer = FrameTransformer(FrameTransformerCfg(
       prim_path="{ENV_REGEX_NS}/Robot/base",
       target_frames=[
           FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/sensor_link"),
       ],
   ))
   sensor_pos = frame_transformer.data.target_pos_w
   sensor_quat = frame_transformer.data.target_quat_w


Articulation Joint Wrench Data Moved to ``JointWrenchSensor``
-------------------------------------------------------------

The ``ArticulationData.body_incoming_joint_wrench_b`` property has been removed. In
Isaac Lab 3.0, incoming joint reaction wrenches are exposed through
:class:`~isaaclab.sensors.JointWrenchSensor`, which has PhysX and Newton backend
implementations and returns separate force [N] and torque [N·m] buffers.
The sensor reports wrenches in the child-side incoming joint frame, with torque
referenced at the child-side joint anchor.

**Before (Isaac Lab 2.x):**

.. code-block:: python

   wrench_b = robot.data.body_incoming_joint_wrench_b.torch[:, body_ids]

**After (Isaac Lab 3.x):**

.. code-block:: python

   import torch
   from isaaclab.scene import InteractiveSceneCfg
   from isaaclab.sensors import JointWrenchSensorCfg

   class MySceneCfg(InteractiveSceneCfg):
       robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
       joint_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")

   sensor = env.scene.sensors["joint_wrench"]
   data = sensor.data
   wrench_j = torch.cat(
       (
           data.force.torch[:, body_ids],
           data.torque.torch[:, body_ids],
       ),
       dim=-1,
   )

Use :attr:`~isaaclab.sensors.BaseJointWrenchSensor.body_names` or
:meth:`~isaaclab.sensors.BaseJointWrenchSensor.find_bodies` to map sensor entries to
articulation body names. PhysX reports one entry for every link, including the articulation
root link. Newton reports the child bodies of reportable incoming joints.

For manager-based environments, update observations that used the articulation data property to
depend on the joint-wrench sensor instead:

.. code-block:: python

   import isaaclab.envs.mdp as mdp
   from isaaclab.managers import SceneEntityCfg
   from isaaclab.managers import ObservationTermCfg as ObsTerm
   from isaaclab.scene import InteractiveSceneCfg
   from isaaclab.sensors import JointWrenchSensorCfg

   class MySceneCfg(InteractiveSceneCfg):
       robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
       joint_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")

   feet_body_forces = ObsTerm(
       func=mdp.body_incoming_wrench,
       params={
           "sensor_cfg": SceneEntityCfg(
               "joint_wrench",
               body_names=["left_foot", "right_foot"],
           )
       },
   )


Multi-Backend Support: PresetCfg Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Isaac Lab 3.0 introduces a **PresetCfg pattern** for writing environment configurations
that work with both the PhysX and Newton backends. Instead of hard-coding a single
physics config, environments declare named configuration variants. The active variant
is selected at launch via a Hydra CLI override.

What is PresetCfg?
------------------

:class:`~isaaclab_tasks.utils.PresetCfg` is a base ``@configclass`` whose typed fields
represent named variants of a configuration section. The field named ``default`` is used
when no CLI override is given. Other fields are named presets selectable with
``presets=<name>`` on the command line:

.. code-block:: python

   from isaaclab_tasks.utils import PresetCfg
   from isaaclab.utils import configclass

   @configclass
   class MyPhysicsCfg(PresetCfg):
       default: PhysxCfg = PhysxCfg(...)   # used when no override is given
       physx:   PhysxCfg = PhysxCfg(...)   # selected by presets=physx
       newton_mjwarp:  NewtonCfg = NewtonCfg(...)  # selected by presets=newton_mjwarp

Selecting a preset at launch
-----------------------------

Pass ``presets=newton_mjwarp`` (or ``presets=physx``) on the CLI to swap the entire config section:

.. code-block:: bash

   # Run with Newton backend
   python train.py task=Isaac-Franka-Cabinet-v0 presets=newton_mjwarp

   # Run with default (PhysX) backend
   python train.py task=Isaac-Franka-Cabinet-v0

Adding Multi-Backend Support to an Environment
-----------------------------------------------

**Step 1 — Physics config**

Replace a plain ``PhysxCfg(...)`` assignment in ``__post_init__`` with a ``PresetCfg``
subclass that carries both a PhysX and a Newton variant.

*Before:*

.. code-block:: python

   def __post_init__(self):
       self.sim.dt = 1 / 60
       self.sim.physics = PhysxCfg(bounce_threshold_velocity=0.2)

*After:*

.. code-block:: python

   from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
   from isaaclab_physx.physics import PhysxCfg
   from isaaclab_tasks.utils import PresetCfg

   @configclass
   class ReachPhysicsCfg(PresetCfg):
       default: PhysxCfg = PhysxCfg(bounce_threshold_velocity=0.2)
       physx:   PhysxCfg = PhysxCfg(bounce_threshold_velocity=0.2)
       newton_mjwarp:  NewtonCfg = NewtonCfg(
           solver_cfg=MJWarpSolverCfg(
               njmax=20, nconmax=20, ls_iterations=20,
               cone="pyramidal", ls_parallel=True,
               integrator="implicitfast", impratio=1,
           ),
           num_substeps=1,
           debug_mode=False,
       )

   # In the env cfg __post_init__:
   def __post_init__(self):
       self.sim.dt = 1 / 60
       self.sim.physics = ReachPhysicsCfg()

Key Newton solver parameters:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Effect
   * - ``njmax``
     - Max constraint rows; set ≥ expected contact count per env
   * - ``nconmax``
     - Max contacts per env
   * - ``ls_iterations``
     - Linear solver iterations (higher = more stable, slower)
   * - ``cone``
     - ``"pyramidal"`` (fast) or ``"elliptic"`` (more accurate)
   * - ``integrator``
     - ``"implicitfast"`` (recommended) or ``"euler"``
   * - ``impratio``
     - Impedance ratio; >1 improves soft contact stability
   * - ``num_substeps``
     - Physics substeps per environment step

**Step 2 — Differentiating Newton and PhysX Configs**

Not all configurations may be the same between Newton and PhysX simulations.
We can provide a Newton-specific config such as:

.. code-block:: python

   @configclass
   class EventCfg:
       """Full event config (PhysX-compatible)."""
       robot_physics_material = EventTerm(
           func=mdp.randomize_rigid_body_material,
           mode="startup",
           params={...},
       )
       reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
       reset_robot_joints = EventTerm(
           func=mdp.reset_joints_by_offset, mode="reset", params={...}
       )


   @configclass
   class _EnvNewtonEventCfg:
       """Newton-compatible events."""
       reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
       reset_robot_joints = EventTerm(
           func=mdp.reset_joints_by_offset, mode="reset", params={...}
       )


   @configclass
   class EnvEventCfg(PresetCfg):
       default: EventCfg = EventCfg()
       physx:   EventCfg = EventCfg()
       newton_mjwarp:  _EnvNewtonEventCfg = _EnvNewtonEventCfg()

Then change the ``events`` field in your env cfg from ``EventCfg`` to ``EnvEventCfg``:

.. code-block:: python

   @configclass
   class MyEnvCfg(ManagerBasedRLEnvCfg):
       events: EnvEventCfg = EnvEventCfg()  # was: EventCfg = EventCfg()


RigidObjectCollection API Renaming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~isaaclab_physx.assets.RigidObjectCollection` and
:class:`~isaaclab_physx.assets.RigidObjectCollectionData` classes have undergone an API rename
to provide consistency with other asset classes. The ``object_*`` naming convention has been
deprecated in favor of ``body_*``.


Method Renames
--------------

The following methods have been renamed. The old methods are deprecated and will be removed in a
future release:

+------------------------------------------+------------------------------------------+
| Deprecated (2.x)                         | New (3.0)                                |
+==========================================+==========================================+
| ``write_object_state_to_sim()``          | ``write_body_state_to_sim()``            |
+------------------------------------------+------------------------------------------+
| ``write_object_link_state_to_sim()``     | ``write_body_link_state_to_sim()``       |
+------------------------------------------+------------------------------------------+
| ``write_object_pose_to_sim()``           | ``write_body_pose_to_sim()``             |
+------------------------------------------+------------------------------------------+
| ``write_object_link_pose_to_sim()``      | ``write_body_link_pose_to_sim()``        |
+------------------------------------------+------------------------------------------+
| ``write_object_com_pose_to_sim()``       | ``write_body_com_pose_to_sim()``         |
+------------------------------------------+------------------------------------------+
| ``write_object_velocity_to_sim()``       | ``write_body_com_velocity_to_sim()``     |
+------------------------------------------+------------------------------------------+
| ``write_object_com_velocity_to_sim()``   | ``write_body_com_velocity_to_sim()``     |
+------------------------------------------+------------------------------------------+
| ``write_object_link_velocity_to_sim()``  | ``write_body_link_velocity_to_sim()``    |
+------------------------------------------+------------------------------------------+
| ``find_objects()``                       | ``find_bodies()``                        |
+------------------------------------------+------------------------------------------+


Property Renames (Data Class)
-----------------------------

The following properties on :class:`~isaaclab_physx.assets.RigidObjectCollectionData` have been
renamed. The old properties are deprecated and will be removed in a future release:

+------------------------------------------+------------------------------------------+
| Deprecated (2.x)                         | New (3.0)                                |
+==========================================+==========================================+
| ``default_object_state``                 | ``default_body_state``                   |
+------------------------------------------+------------------------------------------+
| ``object_names``                         | ``body_names``                           |
+------------------------------------------+------------------------------------------+
| ``object_link_pose_w``                   | ``body_link_pose_w``                     |
+------------------------------------------+------------------------------------------+
| ``object_link_vel_w``                    | ``body_link_vel_w``                      |
+------------------------------------------+------------------------------------------+
| ``object_com_pose_w``                    | ``body_com_pose_w``                      |
+------------------------------------------+------------------------------------------+
| ``object_com_vel_w``                     | ``body_com_vel_w``                       |
+------------------------------------------+------------------------------------------+
| ``object_state_w``                       | ``body_state_w``                         |
+------------------------------------------+------------------------------------------+
| ``object_link_state_w``                  | ``body_link_state_w``                    |
+------------------------------------------+------------------------------------------+
| ``object_com_state_w``                   | ``body_com_state_w``                     |
+------------------------------------------+------------------------------------------+
| ``object_com_acc_w``                     | ``body_com_acc_w``                       |
+------------------------------------------+------------------------------------------+
| ``object_com_pose_b``                    | ``body_com_pose_b``                      |
+------------------------------------------+------------------------------------------+
| ``object_link_pos_w``                    | ``body_link_pos_w``                      |
+------------------------------------------+------------------------------------------+
| ``object_link_quat_w``                   | ``body_link_quat_w``                     |
+------------------------------------------+------------------------------------------+
| ``object_link_lin_vel_w``                | ``body_link_lin_vel_w``                  |
+------------------------------------------+------------------------------------------+
| ``object_link_ang_vel_w``                | ``body_link_ang_vel_w``                  |
+------------------------------------------+------------------------------------------+
| ``object_com_pos_w``                     | ``body_com_pos_w``                       |
+------------------------------------------+------------------------------------------+
| ``object_com_quat_w``                    | ``body_com_quat_w``                      |
+------------------------------------------+------------------------------------------+
| ``object_com_lin_vel_w``                 | ``body_com_lin_vel_w``                   |
+------------------------------------------+------------------------------------------+
| ``object_com_ang_vel_w``                 | ``body_com_ang_vel_w``                   |
+------------------------------------------+------------------------------------------+
| ``object_com_lin_acc_w``                 | ``body_com_lin_acc_w``                   |
+------------------------------------------+------------------------------------------+
| ``object_com_ang_acc_w``                 | ``body_com_ang_acc_w``                   |
+------------------------------------------+------------------------------------------+
| ``object_com_pos_b``                     | ``body_com_pos_b``                       |
+------------------------------------------+------------------------------------------+
| ``object_com_quat_b``                    | ``body_com_quat_b``                      |
+------------------------------------------+------------------------------------------+
| ``object_link_lin_vel_b``                | ``body_link_lin_vel_b``                  |
+------------------------------------------+------------------------------------------+
| ``object_link_ang_vel_b``                | ``body_link_ang_vel_b``                  |
+------------------------------------------+------------------------------------------+
| ``object_com_lin_vel_b``                 | ``body_com_lin_vel_b``                   |
+------------------------------------------+------------------------------------------+
| ``object_com_ang_vel_b``                 | ``body_com_ang_vel_b``                   |
+------------------------------------------+------------------------------------------+
| ``object_pose_w``                        | ``body_pose_w``                          |
+------------------------------------------+------------------------------------------+
| ``object_pos_w``                         | ``body_pos_w``                           |
+------------------------------------------+------------------------------------------+
| ``object_quat_w``                        | ``body_quat_w``                          |
+------------------------------------------+------------------------------------------+
| ``object_vel_w``                         | ``body_vel_w``                           |
+------------------------------------------+------------------------------------------+
| ``object_lin_vel_w``                     | ``body_lin_vel_w``                       |
+------------------------------------------+------------------------------------------+
| ``object_ang_vel_w``                     | ``body_ang_vel_w``                       |
+------------------------------------------+------------------------------------------+
| ``object_lin_vel_b``                     | ``body_lin_vel_b``                       |
+------------------------------------------+------------------------------------------+
| ``object_ang_vel_b``                     | ``body_ang_vel_b``                       |
+------------------------------------------+------------------------------------------+
| ``object_acc_w``                         | ``body_acc_w``                           |
+------------------------------------------+------------------------------------------+
| ``object_lin_acc_w``                     | ``body_lin_acc_w``                       |
+------------------------------------------+------------------------------------------+
| ``object_ang_acc_w``                     | ``body_ang_acc_w``                       |
+------------------------------------------+------------------------------------------+

.. note::

   All deprecated methods and properties will issue a deprecation warning when used. Your existing
   code will continue to work, but you should migrate to the new API to avoid issues in future releases.


Migration Example
-----------------

Here's a complete example showing how to update your code:

**Before (Isaac Lab 2.x):**

.. code-block:: python

   from isaaclab.assets import DeformableObject, DeformableObjectCfg
   from isaaclab.assets import SurfaceGripper, SurfaceGripperCfg
   from isaaclab.assets import RigidObjectCollection

   # Using deprecated root_physx_view
   robot = scene["robot"]
   masses = robot.root_physx_view.get_masses()

   # Using deprecated object_* API
   collection = scene["object_collection"]
   poses = collection.data.object_pose_w
   collection.write_object_state_to_sim(state, env_ids=env_ids, object_ids=object_ids)

**After (Isaac Lab 3.0):**

.. code-block:: python

   from isaaclab_physx.assets import DeformableObject, DeformableObjectCfg
   from isaaclab_physx.assets import SurfaceGripper, SurfaceGripperCfg
   from isaaclab.assets import RigidObjectCollection  # unchanged

   # Using new root_view property
   robot = scene["robot"]
   masses = robot.root_view.get_masses()

   # Using new body_* API
   collection = scene["object_collection"]
   poses = collection.data.body_pose_w
   collection.write_body_state_to_sim(state, env_ids=env_ids, body_ids=object_ids)


Quaternion Format
~~~~~~~~~~~~~~~~~

**The quaternion format changed from WXYZ to XYZW.**

+------------------+----------------------------------+----------------------------------+
| Component        | Old Format (WXYZ)                | New Format (XYZW)                |
+==================+==================================+==================================+
| Order            | ``(w, x, y, z)``                 | ``(x, y, z, w)``                 |
+------------------+----------------------------------+----------------------------------+
| Identity         | ``(1.0, 0.0, 0.0, 0.0)``         | ``(0.0, 0.0, 0.0, 1.0)``         |
+------------------+----------------------------------+----------------------------------+


Why This Change?
----------------

The new XYZW format aligns with:

- **Warp**: NVIDIA's spatial computing framework
- **PhysX**: PhysX physics engine
- **Newton**: Newton multi-solver framework

This alignment removes the need for internal quaternion conversions, making the code simpler,
faster, and less error-prone.


What You Need to Update
-----------------------

Any hard-coded quaternion values in your code need to be converted from WXYZ to XYZW.
This includes:

1. **Configuration files** - ``rot`` parameters in asset configs
2. **Task definitions** - Goal poses, initial states
3. **Controller parameters** - Target orientations
4. **Documentation** - Code examples with quaternions

Also, if you were relying on the :func:`~isaaclab.utils.math.convert_quat` function to convert quaternions, this should
no longer be needed. (This would happen if you were pulling values from the views directly.)

Example: Updating Asset Configuration
-------------------------------------

**Before (WXYZ):**

.. code-block:: python

   from isaaclab.assets import AssetBaseCfg

   cfg = AssetBaseCfg(
       init_state=AssetBaseCfg.InitialStateCfg(
           pos=(0.0, 0.0, 0.5),
           rot=(1.0, 0.0, 0.0, 0.0),  # OLD: w, x, y, z
       ),
   )

**After (XYZW):**

.. code-block:: python

   from isaaclab.assets import AssetBaseCfg

   cfg = AssetBaseCfg(
       init_state=AssetBaseCfg.InitialStateCfg(
           pos=(0.0, 0.0, 0.5),
           rot=(0.0, 0.0, 0.0, 1.0),  # NEW: x, y, z, w
       ),
   )


Using the Quaternion Finder Tool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide a tool to help you find and fix quaternions in your codebase automatically. This is not a bulletproof tool,
but it should help you find most of the quaternions that need to be updated. You *should* review the results manually.

.. warning::
  Do not run the tool on the whole codebase! If you run the tool on our own packages (isaaclab, or isaaclab_tasks for
  instance) it will find all the quaternions that we already converted. This tool is only meant to be used on your own
  codebase with no overlap with our own packages.

Finding Quaternions
-------------------

Run the tool to scan your code for potential quaternions:

.. code-block:: bash

   # Scan the 'source' directory (default)
   python scripts/tools/find_quaternions.py

   # Scan a specific path
   python scripts/tools/find_quaternions.py --path my_project/

   # Compare against a different branch
   python scripts/tools/find_quaternions.py --base develop

.. tip::
  We recommend always running the tool with a custom base branch *and* a specific path.


The tool will show you:

- Quaternions that haven't been updated (marked as ``UNCHANGED``)
- Whether each looks like a WXYZ identity quaternion (``WXYZ_IDENTITY``)
- Whether the format is likely WXYZ (``LIKELY_WXYZ``)


Understanding the Output
------------------------

.. code-block:: text

   my_project/robot_cfg.py:42:8 ⚠ UNCHANGED [WXYZ_IDENTITY]
     Values: [1.0, 0.0, 0.0, 0.0]
     Source: rot=(1.0, 0.0, 0.0, 0.0),

This tells you:

- **File and line**: ``my_project/robot_cfg.py:42``
- **Status**: ``UNCHANGED`` means this line hasn't been modified yet
- **Flag**: ``WXYZ_IDENTITY`` means it's the identity quaternion in old WXYZ format
- **Values**: The actual quaternion values found
- **Source**: The line of code for context


Filtering Results
-----------------

Focus on specific types of quaternions:

.. code-block:: bash

   # Only show identity quaternions [1, 0, 0, 0]
   python scripts/tools/find_quaternions.py --check-identity

   # Only show quaternions likely in WXYZ format
   python scripts/tools/find_quaternions.py --likely-wxyz

   # Show ALL potential quaternions (ignore format heuristics)
   python scripts/tools/find_quaternions.py --all-quats


Fixing Quaternions Automatically
--------------------------------

The tool can automatically convert quaternions from WXYZ to XYZW:

.. code-block:: bash

   # Interactive mode: prompts before each fix
   python scripts/tools/find_quaternions.py --fix

   # Only fix identity quaternions (safest option)
   python scripts/tools/find_quaternions.py --fix-identity-only

   # Preview changes without applying them
   python scripts/tools/find_quaternions.py --fix --dry-run

   # Apply all fixes without prompting
   python scripts/tools/find_quaternions.py --fix --force


Interactive Fix Example
-----------------------

When running with ``--fix``, you'll see something like:

.. code-block:: text

   ────────────────────────────────────────────────────────────────────────────────
   📍 my_project/robot_cfg.py:42 [WXYZ_IDENTITY]
   ────────────────────────────────────────────────────────────────────────────────
        40 |     init_state=AssetBaseCfg.InitialStateCfg(
        41 |         pos=(0.0, 0.0, 0.5),
   >>>  42 |         rot=(1.0, 0.0, 0.0, 0.0),
        43 |     ),
        44 | )
   ────────────────────────────────────────────────────────────────────────────────
     Change: [1.0, 0.0, 0.0, 0.0] → [0.0, 0.0, 0.0, 1.0]
     Result: rot=(0.0, 0.0, 0.0, 1.0),
   Apply this fix? [Y/n/a/q]:

Options:

- **Y** (yes): Apply this fix
- **n** (no): Skip this one
- **a** (all): Apply all remaining fixes without asking
- **q** (quit): Stop fixing


How the Tool Works
------------------

The tool uses several techniques to find quaternions:

1. **Python files**: Parses the code using AST (Abstract Syntax Tree) to find
   4-element tuples and lists with numeric values.

2. **JSON files**: Uses regex to find 4-element arrays.

3. **RST documentation**: Searches for quaternion-like patterns in docs.

To identify if something is a quaternion, the tool checks:

- Is it exactly 4 numeric values?
- Does the sum of squares ≈ 1? (unit quaternion property)
- Does it match known patterns like identity quaternions?

To determine if it's in WXYZ format:

- Is the first value 1.0 and rest are 0? (WXYZ identity)
- Is the first value a common cos(θ/2) value like 0.707, 0.866, etc.?
- Is the pattern consistent with first-element being the scalar part?


Best Practices for Migration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Start with a clean git state** - Commit your work before running fixes.

2. **Run the tool first without ``--fix``** - Review what will be changed.

3. **Fix identity quaternions first** - They're the most common and safest:

   .. code-block:: bash

      python scripts/tools/find_quaternions.py --fix-identity-only

4. **Review non-identity quaternions manually** - Some 4-element lists might
   not be quaternions (e.g., RGBA colors, bounding boxes).

5. **Test your code** - Run your simulations to verify everything works correctly.

6. **Check documentation** - Update any docs or comments that mention quaternion format.


Using the Runtime Quaternion Access Detector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The quaternion finder tool above covers hard-coded values in source files,
but it cannot see quaternions that are *read* from asset/sensor data at
runtime. For those, Isaac Lab ships a
runtime detector hook on :class:`~isaaclab.utils.warp.ProxyArray` that flags
every ``.torch`` access on a ``wp.quatf``-typed property and points at the
exact call site. Use it after the source-level migration to catch the cases
the finder tool can't reach.

Enable it by setting an environment variable before launching your script:

.. code-block:: bash

   export WARN_ON_TORCH_QUATF_ACCESS=1
   ./isaaclab.sh -p my_script.py

Every read of ``.torch`` on a ``ProxyArray`` whose underlying ``wp.array`` has
dtype ``wp.quatf`` then emits a :class:`UserWarning` with the message:

.. code-block:: text

   Reading .torch on a wp.quatf-typed ProxyArray. The Isaac Lab quaternion
   convention changed from (w, x, y, z) in 2.x to (x, y, z, w) in 3.x. If
   your code assumes the old order, this is likely the source of incorrect
   rotations. Unset WARN_ON_TORCH_QUATF_ACCESS to silence this warning.

The warning's traceback points at the exact line that performed the access
(via ``stacklevel=2``), so you can walk through the matches in your code and
confirm each one uses the new ``(x, y, z, w)`` order.

Typical workflow:

1. Run a representative scene or task with the env var set.
2. Triage every warning location — check whether the call site assumes
   ``(w, x, y, z)`` (Lab 2.x) or ``(x, y, z, w)`` (Lab 3.x).
3. Migrate the call sites that still expect the old order.
4. Re-run with the env var still set; the warnings should be gone (or only
   come from intentionally-handled call sites).
5. Unset the env var for production runs — the detector adds an
   ``os.environ`` lookup per ``.torch`` access, which is cheap but not free.

The detector covers only ``ProxyArray.torch`` reads. Direct accesses on the
underlying ``wp.array`` (via ``ProxyArray.warp``) are not flagged, because
warp uses ``(x, y, z, w)`` natively and so a warp-side read is unaffected
by the convention change.


API Changes
~~~~~~~~~~~

The ``convert_quat`` function has been removed
----------------------------------------------

Previously, IsaacLab had a utility function to convert between quaternion formats:

.. code-block:: python

   # OLD - No longer needed
   from isaaclab.utils.math import convert_quat
   quat_xyzw = convert_quat(quat_wxyz, "xyzw")

Since everything now uses XYZW natively, this function is no longer needed.
If you were using it, simply remove the conversion calls.


Math utility functions now expect XYZW
--------------------------------------

All quaternion functions in :mod:`isaaclab.utils.math` now expect and return
quaternions in XYZW format:

- :func:`~isaaclab.utils.math.quat_mul`
- :func:`~isaaclab.utils.math.quat_apply`
- :func:`~isaaclab.utils.math.quat_from_euler_xyz`
- :func:`~isaaclab.utils.math.euler_xyz_from_quat`
- :func:`~isaaclab.utils.math.quat_from_matrix`
- :func:`~isaaclab.utils.math.matrix_from_quat`
- And all other quaternion utilities


ProxyArray Backend for Asset and Sensor Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All ``.data.*`` properties on asset and sensor classes now return
:class:`~isaaclab.utils.warp.ProxyArray` instead of ``torch.Tensor``. ``ProxyArray`` wraps
the underlying ``wp.array`` and exposes explicit ``.torch`` and ``.warp`` accessors. This
change applies to all asset classes (:class:`~isaaclab.assets.Articulation`,
:class:`~isaaclab.assets.RigidObject`, :class:`~isaaclab.assets.RigidObjectCollection`,
:class:`~isaaclab_physx.assets.DeformableObject`) and all sensor classes
(:class:`~isaaclab_physx.sensors.ContactSensor`, :class:`~isaaclab_physx.sensors.Imu`,
:class:`~isaaclab_physx.sensors.Pva`, :class:`~isaaclab_physx.sensors.FrameTransformer`).

To use a data property as a ``torch.Tensor``, append ``.torch``:

.. code-block:: python

   # Before (Isaac Lab 2.x)
   root_pos = robot.data.root_pos_w             # torch.Tensor
   joint_pos = robot.data.joint_pos              # torch.Tensor
   contact_forces = sensor.data.net_forces_w     # torch.Tensor

   # After (Isaac Lab 3.x)
   root_pos = robot.data.root_pos_w              # ProxyArray
   joint_pos = robot.data.joint_pos              # ProxyArray
   contact_forces = sensor.data.net_forces_w     # ProxyArray

   # To use with torch operations, access .torch
   root_pos_torch = robot.data.root_pos_w.torch        # torch.Tensor
   joint_pos_torch = robot.data.joint_pos.torch        # torch.Tensor
   contact_torch = sensor.data.net_forces_w.torch      # torch.Tensor

Common patterns that need updating:

.. code-block:: python

   # Cloning data
   # Before:
   pos = robot.data.root_pos_w.clone()
   # After:
   pos = robot.data.root_pos_w.torch.clone()

   # Creating zero tensors with matching shape
   # Before:
   zeros = torch.zeros_like(robot.data.root_pos_w)
   # After:
   zeros = torch.zeros_like(robot.data.root_pos_w.torch)

   # Assertions in tests
   # Before:
   torch.testing.assert_close(robot.data.root_pos_w, expected)
   # After:
   torch.testing.assert_close(robot.data.root_pos_w.torch, expected)

.. list-table:: Affected classes
   :header-rows: 1
   :widths: 40 60

   * - Class
     - Package
   * - :class:`~isaaclab.assets.Articulation`
     - ``isaaclab`` / ``isaaclab_physx``
   * - :class:`~isaaclab.assets.RigidObject`
     - ``isaaclab`` / ``isaaclab_physx``
   * - :class:`~isaaclab.assets.RigidObjectCollection`
     - ``isaaclab`` / ``isaaclab_physx``
   * - :class:`~isaaclab_physx.assets.DeformableObject`
     - ``isaaclab_physx``
   * - :class:`~isaaclab_physx.sensors.ContactSensor`
     - ``isaaclab_physx``
   * - :class:`~isaaclab_physx.sensors.Imu`
     - ``isaaclab_physx``
   * - :class:`~isaaclab_physx.sensors.Pva`
     - ``isaaclab_physx``
   * - :class:`~isaaclab_physx.sensors.FrameTransformer`
     - ``isaaclab_physx``
   * - :class:`~isaaclab.sensors.RayCaster`
     - ``isaaclab``
   * - :class:`~isaaclab.sensors.RayCasterCamera`
     - ``isaaclab``
   * - :class:`~isaaclab.sensors.MultiMeshRayCaster`
     - ``isaaclab``
   * - :class:`~isaaclab.sensors.MultiMeshRayCasterCamera`
     - ``isaaclab``

.. note::

   ``wp.to_torch(proxy_array)`` is temporarily supported by a compatibility shim. It returns
   the same zero-copy tensor as ``proxy_array.torch`` and emits a one-time
   ``DeprecationWarning``. This shim exists for older migration code and will be removed in a
   future release; prefer ``.torch`` in new code.


Ray Caster Warp Backend
~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~isaaclab.sensors.RayCaster`, :class:`~isaaclab.sensors.RayCasterCamera`,
:class:`~isaaclab.sensors.MultiMeshRayCaster`, and
:class:`~isaaclab.sensors.MultiMeshRayCasterCamera` sensors have been transitioned from a
PyTorch/USD-based backend to a native Warp kernel pipeline. This improves performance by
eliminating per-step tensor allocations and torch-to-warp conversions, but introduces several
breaking changes.


RayCasterData Return Types
--------------------------

The :attr:`~isaaclab.sensors.RayCasterData.pos_w`,
:attr:`~isaaclab.sensors.RayCasterData.quat_w`, and
:attr:`~isaaclab.sensors.RayCasterData.ray_hits_w` properties now return
:class:`~isaaclab.utils.warp.ProxyArray` instead of ``torch.Tensor``. This follows the same
pattern as the general ProxyArray backend migration described above.

.. code-block:: python

   # Before (Isaac Lab 2.x)
   ray_hits = ray_caster.data.ray_hits_w        # torch.Tensor
   sensor_pos = ray_caster.data.pos_w            # torch.Tensor

   # After (Isaac Lab 3.x)
   ray_hits = ray_caster.data.ray_hits_w         # ProxyArray
   sensor_pos = ray_caster.data.pos_w            # ProxyArray

   # To use with torch operations, access .torch
   ray_hits_torch = ray_caster.data.ray_hits_w.torch
   sensor_pos_torch = ray_caster.data.pos_w.torch


Ray Alignment Configuration
----------------------------

The ``attach_yaw_only`` boolean parameter on :class:`~isaaclab.sensors.RayCasterCfg` has been
deprecated in favor of the new ``ray_alignment`` parameter, which accepts one of three string
values:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Old (2.x)
     - New (3.0)
     - Behavior
   * - ``attach_yaw_only=False``
     - ``ray_alignment="base"``
     - Rays follow the full sensor orientation.
   * - ``attach_yaw_only=True``
     - ``ray_alignment="yaw"``
     - Rays follow only the yaw component of the sensor orientation.
   * - *(not available)*
     - ``ray_alignment="world"``
     - Rays are always cast in the world frame (no rotation applied).

.. code-block:: python

   # Before (Isaac Lab 2.x)
   cfg = RayCasterCfg(attach_yaw_only=True, ...)

   # After (Isaac Lab 3.x)
   cfg = RayCasterCfg(ray_alignment="yaw", ...)


Raycasting Kernel Signature Change
-----------------------------------

The :func:`~isaaclab.utils.warp.kernels.raycast_dynamic_meshes_kernel` Warp kernel now requires
an ``env_mask`` parameter as its first argument. This is a ``wp.array(dtype=wp.bool)`` that
controls which environments are updated. The public Python wrapper
:func:`~isaaclab.utils.warp.ops.raycast_dynamic_meshes` has been updated to inject an all-True
mask automatically, so code using the wrapper is unaffected.

If you call the kernel directly, update your launch call:

.. code-block:: python

   import warp as wp

   # Before (Isaac Lab 2.x)
   wp.launch(
       raycast_dynamic_meshes_kernel,
       dim=(num_meshes, num_envs, num_rays),
       inputs=[ray_starts, ray_directions, mesh_ids, ...],
   )

   # After (Isaac Lab 3.x) -- env_mask is now the first input
   env_mask = wp.ones(num_envs, dtype=wp.bool, device=device)
   wp.launch(
       raycast_dynamic_meshes_kernel,
       dim=(num_meshes, num_envs, num_rays),
       inputs=[env_mask, ray_starts, ray_directions, mesh_ids, ...],
   )


RayCaster.meshes Cache Key
--------------------------

The :attr:`~isaaclab.sensors.RayCaster.meshes` class variable, which caches warp meshes across
all :class:`~isaaclab.sensors.RayCaster` instances, is now keyed by ``(prim_path, device)`` tuples
instead of by ``prim_path`` alone. This prevents a mesh that was built on one device (e.g. CPU)
from being reused by a sensor running on a different device (e.g. CUDA), which caused illegal
memory accesses on systems without unified memory.

Code that reads or writes this cache directly must update both the type annotation and the key:

.. code-block:: python

   # Before (Isaac Lab 2.x)
   meshes: ClassVar[dict[str, wp.Mesh]] = {}
   wp_mesh = RayCaster.meshes[prim_path]

   # After (Isaac Lab 3.x)
   meshes: ClassVar[dict[tuple[str, str], wp.Mesh]] = {}
   wp_mesh = RayCaster.meshes[(prim_path, device)]


Write Method Index/Mask Split
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All asset write methods have been split into two explicit variants:

- ``write_*_to_sim_index(data, env_ids)`` — accepts partial data for a sparse set of
  environment indices. The ``data`` tensor has shape ``(len(env_ids), ...)``.
- ``write_*_to_sim_mask(data, env_mask)`` — accepts full data for all environments with a
  boolean mask selecting which environments to update. The ``data`` tensor has shape
  ``(num_envs, ...)``.

The previous ``write_*_to_sim(data, env_ids)`` methods have been removed.

.. code-block:: python

   # Before (Isaac Lab 2.x)
   robot.write_root_pose_to_sim(pose_data, env_ids)

   # After (Isaac Lab 3.x) — indexed variant (partial data)
   robot.write_root_pose_to_sim_index(root_pose=pose_data, env_ids=env_ids)

   # After (Isaac Lab 3.x) — mask variant (full data, boolean mask)
   robot.write_root_pose_to_sim_mask(root_pose=pose_data, env_mask=env_mask)

.. list-table:: Affected write methods (RigidObject / Articulation)
   :header-rows: 1
   :widths: 50 50

   * - Old method
     - New methods
   * - ``write_root_pose_to_sim``
     - ``write_root_pose_to_sim_index`` / ``write_root_pose_to_sim_mask``
   * - ``write_root_link_pose_to_sim``
     - ``write_root_link_pose_to_sim_index`` / ``write_root_link_pose_to_sim_mask``
   * - ``write_root_com_pose_to_sim``
     - ``write_root_com_pose_to_sim_index`` / ``write_root_com_pose_to_sim_mask``
   * - ``write_root_velocity_to_sim``
     - ``write_root_velocity_to_sim_index`` / ``write_root_velocity_to_sim_mask``
   * - ``write_root_com_velocity_to_sim``
     - ``write_root_com_velocity_to_sim_index`` / ``write_root_com_velocity_to_sim_mask``
   * - ``write_root_link_velocity_to_sim``
     - ``write_root_link_velocity_to_sim_index`` / ``write_root_link_velocity_to_sim_mask``

.. list-table:: Additional Articulation-specific write methods
   :header-rows: 1
   :widths: 50 50

   * - Old method
     - New methods
   * - ``write_joint_position_to_sim``
     - ``write_joint_position_to_sim_index`` / ``write_joint_position_to_sim_mask``
   * - ``write_joint_velocity_to_sim``
     - ``write_joint_velocity_to_sim_index`` / ``write_joint_velocity_to_sim_mask``
   * - ``write_joint_stiffness_to_sim``
     - ``write_joint_stiffness_to_sim_index`` / ``write_joint_stiffness_to_sim_mask``
   * - ``write_joint_damping_to_sim``
     - ``write_joint_damping_to_sim_index`` / ``write_joint_damping_to_sim_mask``
   * - ``write_joint_position_limit_to_sim``
     - ``write_joint_position_limit_to_sim_index`` / ``write_joint_position_limit_to_sim_mask``
   * - ``write_joint_velocity_limit_to_sim``
     - ``write_joint_velocity_limit_to_sim_index`` / ``write_joint_velocity_limit_to_sim_mask``
   * - ``write_joint_effort_limit_to_sim``
     - ``write_joint_effort_limit_to_sim_index`` / ``write_joint_effort_limit_to_sim_mask``
   * - ``write_joint_armature_to_sim``
     - ``write_joint_armature_to_sim_index`` / ``write_joint_armature_to_sim_mask``
   * - ``write_joint_friction_coefficient_to_sim``
     - ``write_joint_friction_coefficient_to_sim_index`` / ``write_joint_friction_coefficient_to_sim_mask``

.. list-table:: RigidObjectCollection write methods
   :header-rows: 1
   :widths: 50 50

   * - Old method
     - New methods
   * - ``write_body_pose_to_sim``
     - ``write_body_pose_to_sim_index`` / ``write_body_pose_to_sim_mask``
   * - ``write_body_link_pose_to_sim``
     - ``write_body_link_pose_to_sim_index`` / ``write_body_link_pose_to_sim_mask``
   * - ``write_body_com_pose_to_sim``
     - ``write_body_com_pose_to_sim_index`` / ``write_body_com_pose_to_sim_mask``
   * - ``write_body_velocity_to_sim``
     - ``write_body_velocity_to_sim_index`` / ``write_body_velocity_to_sim_mask``
   * - ``write_body_com_velocity_to_sim``
     - ``write_body_com_velocity_to_sim_index`` / ``write_body_com_velocity_to_sim_mask``
   * - ``write_body_link_velocity_to_sim``
     - ``write_body_link_velocity_to_sim_index`` / ``write_body_link_velocity_to_sim_mask``


TimestampedBufferWarp
~~~~~~~~~~~~~~~~~~~~~

If you have custom asset or sensor data classes that subclass the Isaac Lab base data classes,
note that internal buffers have changed from :class:`~isaaclab.utils.buffers.TimestampedBuffer`
to :class:`~isaaclab.utils.buffers.TimestampedBufferWarp`. The new class takes ``(shape, device,
wp_dtype)`` as constructor arguments instead of a ``torch.Tensor``:

.. code-block:: python

   import warp as wp
   from isaaclab.utils.buffers import TimestampedBufferWarp

   # Before (Isaac Lab 2.x)
   self._data.root_pos_w = TimestampedBuffer(torch.zeros(num_envs, 3, device=device))

   # After (Isaac Lab 3.x)
   self._data.root_pos_w = TimestampedBufferWarp(
       shape=(num_envs,), device=device, wp_dtype=wp.vec3f
   )


URDF Importer
~~~~~~~~~~~~~

The URDF importer in Isaac Sim was rewritten to version 3.0, using the ``urdf-usd-converter``
library and the ``isaacsim.asset.transformer.rules`` extension to produce structured USD output.
The old C++ binding-based API (using Kit commands ``URDFParseFile``/``URDFImportRobot`` and the
``_urdf`` interface from ``acquire_urdf_interface()``) has been replaced with a new Python-based
pipeline.

The IsaacLab :class:`~sim.converters.UrdfConverter` has been updated to replicate the new
``URDFImporter.import_urdf()`` pipeline, inserting IsaacLab-specific post-processing (fix base,
joint drives, link density) on the intermediate USD stage before the asset transformer
restructures the output.

.. important::

   The previous version-pinning mechanism that locked the URDF importer extension to
   ``isaacsim.asset.importer.urdf-2.4.31`` has been removed. The converter now uses whichever
   version of the extension is available in your Isaac Sim installation.


Deprecated Settings
-------------------

The following :class:`~sim.converters.UrdfConverterCfg` settings are **deprecated** because
the new URDF importer 3.0 no longer supports them. They are kept for backward compatibility
but will log warnings if enabled:

+-----------------------------------------------------------+-----------------------------------------------------+
| Setting                                                   | Notes                                               |
+===========================================================+=====================================================+
| ``convert_mimic_joints_to_normal_joints``                 | No longer supported by the importer.                |
+-----------------------------------------------------------+-----------------------------------------------------+
| ``replace_cylinders_with_capsules``                       | No longer supported by the importer.                |
+-----------------------------------------------------------+-----------------------------------------------------+
| ``root_link_name``                                        | No longer supported by the importer.                |
+-----------------------------------------------------------+-----------------------------------------------------+

.. note::

   The ``merge_fixed_joints`` setting is **still supported**. It is now implemented as a URDF XML
   pre-processing step that runs before the USD conversion. Fixed joints are removed and child
   link elements (visual, collision, inertial) are merged into the parent link with correct
   transform composition.

Additionally, the :class:`~sim.converters.UrdfConverterCfg.JointDriveCfg.NaturalFrequencyGainsCfg`
gains mode is **deprecated**. The ``compute_natural_stiffness`` function that it depended on has
been removed from the importer. If ``NaturalFrequencyGainsCfg`` is used, a
:exc:`DeprecationWarning` is emitted and joint drive gains are left at the values produced by the
URDF importer. Use :class:`~sim.converters.UrdfConverterCfg.JointDriveCfg.PDGainsCfg` instead.

The :attr:`~sim.converters.AssetConverterBaseCfg.make_instanceable` setting from the base class
is also no longer supported and will be ignored. Assets will be made instanceable by default.


Updated CLI Tool
----------------

The ``convert_urdf.py`` script has been updated. The ``usd_file_name`` is now determined
automatically by the importer based on the robot name and cannot be overridden.

**Before (Isaac Lab 2.x):**

.. code-block:: bash

   ./isaaclab.sh -p scripts/tools/convert_urdf.py \
     robot.urdf \
     /output/dir/robot.usd \
     --fix-base \
     --merge-joints

**After (Isaac Lab 3.0):**

.. code-block:: bash

   ./isaaclab.sh -p scripts/tools/convert_urdf.py \
     robot.urdf \
     /output/dir \
     --fix-base \
     --joint-stiffness 100.0 \
     --joint-damping 1.0

.. note::

   The ``--merge-joints`` flag is still accepted and correctly triggers the pre-processing
   step to merge fixed joints.


Updated Python API
------------------

If you use :class:`~sim.converters.UrdfConverter` or :class:`~sim.converters.UrdfConverterCfg`
directly in your code, note the following changes:

1. The ``usd_file_name`` is now set automatically by the converter based on the URDF file name.
   The importer generates output at ``{usd_dir}/{robot_name}/{robot_name}.usda``.

2. The ``make_instanceable`` setting is no longer supported. Assets will be made instanceable
   by default.

3. The ``merge_fixed_joints`` parameter is now implemented as a pre-processing step.

**Before (Isaac Lab 2.x):**

.. code-block:: python

   from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

   cfg = UrdfConverterCfg(
       asset_path="robot.urdf",
       usd_dir="/output/dir",
       usd_file_name="robot.usd",
       fix_base=True,
       merge_fixed_joints=True,
       make_instanceable=True,
       joint_drive=UrdfConverterCfg.JointDriveCfg(
           gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
               stiffness=None,  # use URDF values
               damping=None,
           ),
       ),
   )

**After (Isaac Lab 3.0):**

.. code-block:: python

   from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

   cfg = UrdfConverterCfg(
       asset_path="robot.urdf",
       usd_dir="/output/dir",
       # usd_file_name is determined automatically from the robot name
       fix_base=True,
       merge_fixed_joints=True,  # supported via pre-processing
       joint_drive=UrdfConverterCfg.JointDriveCfg(
           gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
               stiffness=None,  # use URDF values
               damping=None,
           ),
       ),
   )


MJCF Importer
~~~~~~~~~~~~~

The MJCF importer in Isaac Sim was rewritten to use the ``mujoco-usd-converter`` library.
The old C++ binding-based API (using Kit commands ``MJCFCreateAsset``/``MJCFCreateImportConfig``
and the ``ImportConfig`` class) has been replaced with a new pure-Python ``MJCFImporter`` class
and ``MJCFImporterConfig`` dataclass.

.. important::

   The new MJCF importer produces USD assets with **nested rigid bodies** (i.e., ``RigidBodyAPI``
   is applied to each link prim individually) instead of a single articulation root with rigid
   body applied only at the top level. This matches how MuJoCo represents bodies and is
   physically more accurate, but it may affect code that assumes a flat rigid body hierarchy.
   If you have downstream logic that traverses the USD structure of MJCF-imported assets,
   verify that it handles nested rigid body prims correctly.

Removed Settings
----------------

The following :class:`~sim.converters.MjcfConverterCfg` settings have been **removed** because
the new converter handles them automatically based on the MJCF file content:

- ``fix_base`` — base fixedness is now inferred from the MJCF ``<freejoint>`` tag.
- ``link_density`` — density is now read directly from the MJCF model.
- ``import_inertia_tensor`` — inertia tensors are always imported.
- ``import_sites`` — sites are always imported.

The :attr:`~sim.converters.AssetConverterBaseCfg.make_instanceable` setting from the base class
is also no longer supported and will be ignored.


New Settings
------------

The following new settings were added to :class:`~sim.converters.MjcfConverterCfg`:

+-----------------------------------------------------------------+------------------------------------------------------+
| Setting                                                         | Description                                          |
+=================================================================+======================================================+
| :attr:`~sim.converters.MjcfConverterCfg.merge_mesh`             | Merge meshes where possible to optimize the model.   |
+-----------------------------------------------------------------+------------------------------------------------------+
| :attr:`~sim.converters.MjcfConverterCfg.collision_from_visuals` | Generate collision geometry from visuals.            |
+-----------------------------------------------------------------+------------------------------------------------------+
| :attr:`~sim.converters.MjcfConverterCfg.collision_type`         | Type of collision geometry (e.g. ``"default"``,      |
|                                                                 | ``"Convex Hull"``, ``"Convex Decomposition"``).      |
+-----------------------------------------------------------------+------------------------------------------------------+


Renamed Settings
----------------

+------------------------------------------+------------------------------------------+
| Old (2.x)                                | New (3.0)                                |
+==========================================+==========================================+
| ``self_collision``                       | ``self_collision`` (unchanged)           |
+------------------------------------------+------------------------------------------+

.. note::

   The underlying Isaac Sim API renamed ``self_collision`` to ``allow_self_collision``.
   The IsaacLab :class:`~sim.converters.MjcfConverterCfg` keeps using ``self_collision``
   for backward compatibility and maps it to the new name internally.


Updated CLI Tool
----------------

The ``convert_mjcf.py`` script has been updated to match the new importer settings.
Old command-line flags (``--fix-base``, ``--make-instanceable``, ``--import-sites``)
are no longer available.

**Before (Isaac Lab 2.x):**

.. code-block:: bash

   ./isaaclab.sh -p scripts/tools/convert_mjcf.py \
     ../mujoco_menagerie/unitree_h1/h1.xml \
     source/isaaclab_assets/data/Robots/Unitree/h1.usd \
     --import-sites \
     --make-instanceable

**After (Isaac Lab 3.0):**

.. code-block:: bash

   ./isaaclab.sh -p scripts/tools/convert_mjcf.py \
     ../mujoco_menagerie/unitree_h1/h1.xml \
     source/isaaclab_assets/data/Robots/Unitree/h1.usd \
     --merge-mesh \
     --self-collision

New flags: ``--merge-mesh``, ``--collision-from-visuals``, ``--collision-type``, ``--self-collision``.


Updated Python API
------------------

If you use :class:`~sim.converters.MjcfConverter` or :class:`~sim.converters.MjcfConverterCfg`
directly in your code, update your configuration:

**Before (Isaac Lab 2.x):**

.. code-block:: python

   from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

   cfg = MjcfConverterCfg(
       asset_path="robot.xml",
       usd_dir="/output/dir",
       fix_base=True,
       import_sites=True,
       make_instanceable=True,
   )

**After (Isaac Lab 3.0):**

.. code-block:: python

   from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

   cfg = MjcfConverterCfg(
       asset_path="robot.xml",
       usd_dir="/output/dir",
       merge_mesh=True,
       collision_from_visuals=False,
       self_collision=False,
   )


XR Teleoperation: Isaac Teleop Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The native XR teleoperation stack in ``isaaclab.devices.openxr`` has been deprecated and replaced
by `Isaac Teleop <https://github.com/NVIDIA/IsaacTeleop>`_, integrated via the ``isaaclab_teleop``
extension. The ``isaac-teleop-device-plugins`` repository has also been deprecated; all device
plugin support is now in Isaac Teleop.

For full documentation on the new stack, see :ref:`isaac-teleop-feature`.


Installation Requirement
------------------------

Isaac Teleop must now be installed in your Isaac Lab environment:

.. code-block:: bash

   pip install isaacteleop~=1.0 --extra-index-url https://pypi.nvidia.com

See :ref:`install-isaac-teleop` for complete installation instructions.


Import Changes
--------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Deprecated (2.x)
     - New (3.0)
   * - ``from isaaclab.devices.openxr import OpenXRDevice``
     - ``from isaaclab_teleop import IsaacTeleopDevice``
   * - ``from isaaclab.devices.openxr import OpenXRDeviceCfg``
     - ``from isaaclab_teleop import IsaacTeleopCfg``
   * - ``from isaaclab.devices.openxr import XrCfg``
     - ``from isaaclab_teleop import XrCfg``
   * - ``from isaaclab.devices.openxr import ManusVive``
     - ``from isaaclab_teleop import IsaacTeleopDevice`` (with Manus plugin configured)
   * - ``from isaaclab.devices import RetargeterBase``
     - Use Isaac Teleop ``BaseRetargeter`` and pipeline builder pattern
   * - ``from isaaclab.devices.openxr.retargeters import Se3AbsRetargeter``
     - ``from isaacteleop.retargeters import Se3AbsRetargeter``


Environment Configuration Changes
----------------------------------

The ``teleop_devices`` field with ``OpenXRDeviceCfg`` has been replaced by the ``isaac_teleop``
field with ``IsaacTeleopCfg`` and a pipeline builder callable.

**Before (Isaac Lab 2.x):**

.. code-block:: python

   from isaaclab.devices import DevicesCfg, OpenXRDeviceCfg
   from isaaclab.devices.openxr import XrCfg
   from isaaclab.devices.openxr.retargeters import Se3AbsRetargeterCfg, GripperRetargeterCfg

   @configclass
   class MyEnvCfg(ManagerBasedRLEnvCfg):

       xr: XrCfg = XrCfg(anchor_pos=[0.0, 0.0, 0.0])

       teleop_devices: DevicesCfg = field(default_factory=lambda: DevicesCfg(
           handtracking=OpenXRDeviceCfg(
               xr_cfg=None,
               retargeters=[
                   Se3AbsRetargeterCfg(bound_hand=0, zero_out_xy_rotation=True),
                   GripperRetargeterCfg(bound_hand=0),
               ]
           ),
       ))

**After (Isaac Lab 3.0):**

.. code-block:: python

   from isaaclab_teleop import IsaacTeleopCfg, XrCfg

   def _build_pipeline():
       from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
       from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
       from isaacteleop.retargeters import (
           GripperRetargeter, GripperRetargeterConfig,
           Se3AbsRetargeter, Se3RetargeterConfig, TensorReorderer,
       )
       from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

       controllers = ControllersSource(name="controllers")
       hands = HandsSource(name="hands")
       transform = ValueInput("world_T_anchor", TransformMatrix())
       t_controllers = controllers.transformed(transform.output(ValueInput.VALUE))

       se3 = Se3AbsRetargeter(Se3RetargeterConfig(input_device=ControllersSource.RIGHT), name="ee")
       c_se3 = se3.connect({ControllersSource.RIGHT: t_controllers.output(ControllersSource.RIGHT)})

       grip = GripperRetargeter(GripperRetargeterConfig(hand_side="right"), name="grip")
       c_grip = grip.connect({
           ControllersSource.RIGHT: t_controllers.output(ControllersSource.RIGHT),
           HandsSource.RIGHT: hands.output(HandsSource.RIGHT),
       })

       reorder = TensorReorderer(
           input_config={"ee": ["pos_x","pos_y","pos_z","quat_x","quat_y","quat_z","quat_w"],
                         "grip": ["gripper_value"]},
           output_order=["pos_x","pos_y","pos_z","quat_x","quat_y","quat_z","quat_w","gripper_value"],
           name="reorder", input_types={"ee": "array", "grip": "scalar"},
       )
       c_reorder = reorder.connect({"ee": c_se3.output("ee_pose"), "grip": c_grip.output("gripper_command")})
       return OutputCombiner({"action": c_reorder.output("output")})

   @configclass
   class MyEnvCfg(ManagerBasedRLEnvCfg):

       xr: XrCfg = XrCfg(anchor_pos=(0.0, 0.0, 0.0))

       def __post_init__(self):
           super().__post_init__()
           self.isaac_teleop = IsaacTeleopCfg(
               pipeline_builder=_build_pipeline,
               sim_device=self.sim.device,
               xr_cfg=self.xr,
           )


Backward Compatibility
----------------------

The old classes still exist and will issue ``DeprecationWarning`` when used:

* ``isaaclab.devices.openxr.OpenXRDevice`` and ``OpenXRDeviceCfg``
* ``isaaclab.devices.openxr.ManusVive`` and ``ManusViveCfg``
* All retargeters under ``isaaclab.devices.openxr.retargeters``

Deprecated retargeters have been moved to ``isaaclab_teleop.deprecated.openxr.retargeters`` for
compatibility. These will be removed in a future release.


.. _torcharray-migration:

ProxyArray Interop and Temporary Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Asset and sensor data class properties return :class:`~isaaclab.utils.warp.ProxyArray`, a
lightweight wrapper with explicit ``.torch`` and ``.warp`` accessors:

.. code-block:: python

   # BEFORE (2.x) — properties returned torch.Tensor directly
   joint_pos = robot.data.joint_pos          # torch.Tensor
   root_pos = robot.data.root_pos_w          # torch.Tensor

   # AFTER (3.0) — properties return ProxyArray, use .torch for the tensor
   joint_pos = robot.data.joint_pos.torch    # cached zero-copy torch.Tensor
   root_pos = robot.data.root_pos_w.torch    # cached zero-copy torch.Tensor
   joint_pos_warp = robot.data.joint_pos.warp  # the underlying warp.array

**Automatic interop — in many cases, no changes are needed:**

- **Warp kernels:** ``ProxyArray`` implements ``__cuda_array_interface__``, so it can be passed
  directly to ``wp.launch()`` without calling ``.warp``:

  .. code-block:: python

     # Just works — no .warp needed
     wp.launch(my_kernel, inputs=[robot.data.joint_pos], ...)

- **Torch functions:** ``ProxyArray`` implements ``__torch_function__``, so ``torch.*`` operations
  accept it directly. During the deprecation period this emits a one-time warning, but works:

  .. code-block:: python

     # Works (emits DeprecationWarning once, then silent)
     mean_pos = torch.mean(robot.data.joint_pos, dim=1)
     clipped = torch.clamp(robot.data.joint_pos, -3.14, 3.14)

**What to change:**

1. Append ``.torch`` where you need an explicit ``torch.Tensor`` (e.g., for indexing, slicing,
   or passing to non-torch libraries).
2. Warp kernel calls need no changes — ``ProxyArray`` works transparently.
3. If you need the underlying ``warp.array`` (e.g., for ``ptr``, ``strides``), use ``.warp``.
4. Replace legacy ``wp.to_torch(proxy_array)`` calls with ``proxy_array.torch``.

.. note::

   The ``__torch_function__`` bridge and the temporary ``wp.to_torch(proxy_array)`` shim will
   be removed in a future release. We recommend migrating to explicit ``.torch`` access now.

For a complete guide, see :doc:`/source/how-to/proxy_array`.


Migration off Deprecated Isaac Sim APIs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Isaac Sim 6.0, the legacy ``isaacsim.core.*``, ``isaacsim.sensors.*``, and
``isaacsim.robot.wheeled_robots`` Python module paths are **deprecated** in favor of their
``isaacsim.core.experimental.*`` (and ``*.experimental.*``) equivalents. Isaac Lab 3.0 has
been migrated off the deprecated paths so that Isaac Lab continues to load and run when
those modules are removed in a future Isaac Sim release.

This is mostly a transparent change for users — Isaac Lab's own public Python API
(:mod:`isaaclab`, :mod:`isaaclab_physx`, :mod:`isaaclab_tasks`, :mod:`isaaclab_teleop`,
:mod:`isaaclab_mimic`) is unchanged. The migration is only user-visible if you:

1. Import Isaac Sim symbols **directly** in your project, or
2. Maintain a custom Kit experience (``.kit`` file) that lists Isaac Sim extension
   dependencies, or
3. Imported ``SimulationManager`` from ``isaacsim.core.simulation_manager`` in your own
   PhysX-backed code.


Python module renames
---------------------

Update direct imports in your own code as follows. **Where Isaac Lab provides an in-tree
replacement, prefer the Isaac Lab API** over the ``isaacsim.core.experimental.*`` fallback:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Deprecated Isaac Sim path
     - Recommended replacement
   * - ``isaacsim.core.utils.stage``
     - :mod:`isaaclab.sim.utils.stage` (e.g. ``get_current_stage``,
       ``create_new_stage``, ``open_stage``, ``save_stage``, ``close_stage``,
       ``clear_stage``, ``update_stage``, ``use_stage``)
   * - ``isaacsim.core.utils.prims``
     - :mod:`isaaclab.sim.utils.prims` (e.g. ``create_prim``, ``delete_prim``,
       ``change_prim_property``, ``bind_visual_material``,
       ``bind_physics_material``, ``add_usd_reference``)
   * - ``isaacsim.core.utils.queries``
     - :mod:`isaaclab.sim.utils.queries` (e.g. ``find_matching_prims``,
       ``find_matching_prim_paths``, ``get_first_matching_child_prim``)
   * - ``isaacsim.core.utils.transforms``
     - :mod:`isaaclab.sim.utils.transforms`
   * - ``isaacsim.core.utils.semantics``
     - :mod:`isaaclab.sim.utils.semantics`
   * - ``isaacsim.core.utils.extensions.enable_extension``
     - ``isaacsim.core.experimental.utils.app.enable_extension`` (no Isaac Lab equivalent)
   * - ``isaacsim.core.utils.viewports.set_camera_view``
     - ``isaacsim.core.rendering_manager.ViewportManager.set_camera_view`` (or
       ``omni.kit.viewport.utility.camera_state.ViewportCameraState`` for lower-level control)
   * - ``isaacsim.core.prims.XFormPrim`` / ``XFormPrimView``
     - :class:`~isaaclab.sim.views.FrameView` (Isaac Lab in-tree view; see
       :ref:`migrating-to-isaaclab-3-0` ``Renaming of XformPrimView to FrameView`` above).
       For ``Articulation`` / ``RigidPrim`` use ``isaacsim.core.experimental.prims``.
   * - ``isaacsim.core.simulation_manager.SimulationManager``
     - :class:`isaaclab_physx.physics.PhysxManager` (PhysX backend) or
       ``isaaclab_newton.physics.NewtonManager`` (Newton backend); see local-alias
       pattern below.
   * - ``isaacsim.core.cloner``
     - :mod:`isaaclab.cloner` (Isaac Lab in-tree cloner)
   * - ``isaacsim.replicator.mobility_gen``
     - ``isaacsim.replicator.experimental.mobility_gen``
   * - ``isaacsim.sensors.<name>``
     - ``isaacsim.sensors.experimental.<name>``
   * - ``isaacsim.robot.wheeled_robots``
     - ``isaacsim.robot.experimental.wheeled_robots`` (and
       ``isaacsim.robot.wheeled_robots.nodes`` for OmniGraph nodes)

To keep call-site code symmetric across backends when migrating off
``isaacsim.core.simulation_manager.SimulationManager``, use the local-alias pattern:

.. code-block:: python

   from isaaclab_physx.physics import PhysxManager as SimulationManager
   # or, for the Newton backend
   from isaaclab_newton.physics import NewtonManager as SimulationManager


Kit experience (``.kit``) updates
---------------------------------

If you maintain a custom Kit experience derived from one of the Isaac Lab apps under
``apps/``:

* **Stop registering deprecated extension search paths.** The ``extsDeprecated`` search
  path entry has been removed from all stock Isaac Lab Kit experiences (headless,
  rendering, XR variants). Mirror that change in your own experience.
* **Switch explicit Isaac Sim extension dependencies** to the non-deprecated equivalents
  listed above (``isaacsim.core.experimental.*``, ``isaacsim.sensors.experimental.*``,
  ``isaacsim.robot.experimental.wheeled_robots``).
* **Remove unused Isaac Sim extensions that pull in** ``isaacsim.core.api`` — Isaac Lab
  no longer depends on those, and keeping them resurrects the deprecated stack.


``SimulationManager`` is no longer re-exported
----------------------------------------------

Earlier internal previews of this migration briefly exposed
``isaaclab_physx.physics.SimulationManager`` as a public alias of
:class:`~isaaclab_physx.physics.PhysxManager`. **That alias has been removed**; use
:class:`~isaaclab_physx.physics.PhysxManager` directly (with ``as SimulationManager`` at
the import site if you want backend-agnostic call-site code, as shown above).


Retired standalone reproducers
------------------------------

A handful of legacy reproducers under ``source/isaaclab/test/deps/isaacsim`` that
depended on the deprecated Isaac Sim core extensions have been retired:
``check_camera.py``, ``check_floating_base_made_fixed.py``,
``check_legged_robot_clone.py``, ``check_rep_texture_randomizer.py``, and
``check_ref_count.py``. Use :mod:`isaaclab.sim` together with the new
``isaacsim.core.experimental.*`` APIs for the same debugging workflows.


PhysX Tensors API Module Path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recent Isaac Sim releases removed the internal ``impl`` submodule of
``omni.physics.tensors`` and now expose the PhysX Tensor API types
(``ArticulationView``, ``RigidBodyView``, ``SimulationView``, etc.) directly
under ``omni.physics.tensors.api``. Importing from the old path raises
``ModuleNotFoundError: No module named 'omni.physics.tensors.impl'`` at import
time.

Isaac Lab has been updated to import from the new path. Downstream code
(custom assets, sensors, or scripts) that imported from the old path must be
updated:

.. code-block:: python

   # Before (Isaac Lab 2.x / older Isaac Sim)
   import omni.physics.tensors.impl.api as physx

   # After (Isaac Lab 3.x / current Isaac Sim)
   import omni.physics.tensors.api as physx

The class identities are unchanged — only the module path moved. Type hints
referencing the old path (``omni.physics.tensors.impl.api.ArticulationView``)
should be similarly updated to ``omni.physics.tensors.api.ArticulationView``.


Need Help?
~~~~~~~~~~

If you encounter issues during migration:

1. Check the `IsaacLab GitHub Issues <https://github.com/isaac-sim/IsaacLab/issues>`_
2. Review the `CHANGELOG <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/docs/CHANGELOG.rst>`_
3. Join the community on `Discord <https://discord.gg/nvidiaomniverse>`_
