.. _migrating-to-isaaclab-3-0:

Migrating to Isaac Lab 3.0
==========================

.. currentmodule:: isaaclab

Isaac Lab 3.0 introduces a multi-backend architecture that separates simulation backend-specific code
from the core Isaac Lab API. This allows for future support of different physics backends while
maintaining a consistent user-facing API.

This guide covers the main breaking changes and deprecations you need to address when migrating
from Isaac Lab 2.x to Isaac Lab 3.0.


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
- :class:`~isaaclab.sensors.FrameTransformer`, :class:`~isaaclab.sensors.FrameTransformerCfg`, :class:`~isaaclab.sensors.FrameTransformerData`

These sensor classes now use factory patterns that automatically instantiate the appropriate backend
implementation (PhysX by default), maintaining full backward compatibility.

If you need to import the PhysX sensor implementations directly (e.g., for type hints or subclassing),
you can import from ``isaaclab_physx.sensors``:

.. code-block:: python

   # Direct PhysX implementation imports
   from isaaclab_physx.sensors import ContactSensor, ContactSensorData
   from isaaclab_physx.sensors import Imu, ImuData
   from isaaclab_physx.sensors import FrameTransformer, FrameTransformerData


New ``isaaclab_newton`` Extension
---------------------------------

A new extension ``isaaclab_newton`` provides Newton physics backend implementations for:

- :class:`~isaaclab_newton.assets.Articulation` and :class:`~isaaclab_newton.assets.ArticulationData`
- :class:`~isaaclab_newton.assets.RigidObject` and :class:`~isaaclab_newton.assets.RigidObjectData`

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


Sensor Pose Properties Deprecation
----------------------------------

The ``pose_w``, ``pos_w``, and ``quat_w`` properties on :class:`~isaaclab.sensors.ContactSensorData`
and :class:`~isaaclab.sensors.ImuData` are deprecated and will be removed in a future release.

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


``root_physx_view`` Deprecation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``root_physx_view`` property has been deprecated on :class:`~isaaclab.assets.Articulation`,
:class:`~isaaclab.assets.RigidObject`, :class:`~isaaclab.assets.RigidObjectCollection`, and
:class:`~isaaclab_physx.assets.DeformableObject` in favor of the backend-agnostic ``root_view`` property.

+----------------------------------------------+------------------------------------------+
| Deprecated (2.x)                             | New (3.0)                                |
+==============================================+==========================================+
| ``articulation.root_physx_view``             | ``articulation.root_view``               |
+----------------------------------------------+------------------------------------------+
| ``rigid_object.root_physx_view``             | ``rigid_object.root_view``               |
+----------------------------------------------+------------------------------------------+
| ``rigid_object_collection.root_physx_view``  | ``rigid_object_collection.root_view``    |
+----------------------------------------------+------------------------------------------+
| ``deformable_object.root_physx_view``        | ``deformable_object.root_view``          |
+----------------------------------------------+------------------------------------------+

.. note::

   The ``root_view`` property returns the same underlying PhysX view object. This rename is part of
   the multi-backend architecture to provide a consistent API across different physics backends.
   The ``root_physx_view`` property will continue to work but will issue a deprecation warning.


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


Warp Backend for Asset and Sensor Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All ``.data.*`` properties on asset and sensor classes now return ``wp.array`` instead of
``torch.Tensor``. This change applies to all asset classes (:class:`~isaaclab.assets.Articulation`,
:class:`~isaaclab.assets.RigidObject`, :class:`~isaaclab.assets.RigidObjectCollection`,
:class:`~isaaclab_physx.assets.DeformableObject`) and all sensor classes
(:class:`~isaaclab_physx.sensors.ContactSensor`, :class:`~isaaclab_physx.sensors.Imu`,
:class:`~isaaclab_physx.sensors.FrameTransformer`).

To convert back to ``torch.Tensor`` for use with PyTorch operations, wrap the property
access with ``wp.to_torch()``:

.. code-block:: python

   import warp as wp

   # Before (Isaac Lab 2.x)
   root_pos = robot.data.root_pos_w             # torch.Tensor
   joint_pos = robot.data.joint_pos              # torch.Tensor
   contact_forces = sensor.data.net_forces_w     # torch.Tensor

   # After (Isaac Lab 3.x)
   root_pos = robot.data.root_pos_w              # wp.array
   joint_pos = robot.data.joint_pos              # wp.array
   contact_forces = sensor.data.net_forces_w     # wp.array

   # To use with torch operations, wrap with wp.to_torch()
   root_pos_torch = wp.to_torch(robot.data.root_pos_w)        # torch.Tensor
   joint_pos_torch = wp.to_torch(robot.data.joint_pos)        # torch.Tensor
   contact_torch = wp.to_torch(sensor.data.net_forces_w)      # torch.Tensor

Common patterns that need updating:

.. code-block:: python

   # Cloning data
   # Before:
   pos = robot.data.root_pos_w.clone()
   # After:
   pos = wp.to_torch(robot.data.root_pos_w).clone()

   # Creating zero tensors with matching shape
   # Before:
   zeros = torch.zeros_like(robot.data.root_pos_w)
   # After:
   zeros = torch.zeros_like(wp.to_torch(robot.data.root_pos_w))

   # Assertions in tests
   # Before:
   torch.testing.assert_close(robot.data.root_pos_w, expected)
   # After:
   torch.testing.assert_close(wp.to_torch(robot.data.root_pos_w), expected)

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
   * - :class:`~isaaclab_physx.sensors.FrameTransformer`
     - ``isaaclab_physx``

.. note::

   An automated migration tool is provided at ``scripts/tools/wrap_warp_to_torch.py``.
   It scans Python files for ``.data.<property>`` accesses and wraps them with
   ``wp.to_torch()``. Usage:

   .. code-block:: bash

      # Dry run (preview changes)
      python scripts/tools/wrap_warp_to_torch.py path/to/your/code --dry-run

      # Apply changes in-place
      python scripts/tools/wrap_warp_to_torch.py path/to/your/code

   Always review the changes after running the tool, as some accesses (e.g., those
   already passed to warp-native functions) should not be wrapped.


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


Need Help?
~~~~~~~~~~

If you encounter issues during migration:

1. Check the `IsaacLab GitHub Issues <https://github.com/isaac-sim/IsaacLab/issues>`_
2. Review the `CHANGELOG <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/docs/CHANGELOG.rst>`_
3. Join the community on `Discord <https://discord.gg/nvidiaomniverse>`_
