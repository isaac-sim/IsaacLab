:orphan:

.. _isaac-lab-robots:
.. _tutorial-add-new-robot:
.. _how-to-write-articulation-config:


Robot and articulation configuration
====================================

.. currentmodule:: isaaclab

A jointed robot is represented as an articulation in Isaac Lab. This guide covers reusing
an existing robot configuration and authoring a new :class:`~assets.ArticulationCfg`.
The :class:`~assets.ArticulationCfg` is a configuration object that defines the
properties of an :class:`~assets.Articulation` in Isaac Lab.

.. note::

   While we only cover the creation of an :class:`~assets.ArticulationCfg` in this guide,
   the process is similar for creating any other asset configuration object.

Reusing a robot configuration
-----------------------------

Maintained robot configurations live in ``source/isaaclab_assets/isaaclab_assets/robots``.
Import a configuration from ``isaaclab_assets`` and copy it before changing its spawn
properties, initial state, or actuators. Keep project-specific configurations in a Python
module in your own project; they do not need to be added to Isaac Lab.

For example, ``robot_cfg = CARTPOLE_CFG.copy()`` creates an independent configuration.
Use ``robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")`` when adding it to an
:class:`~scene.InteractiveSceneCfg`. The physics backend is selected separately from
this robot configuration, as explained below.

We will use the Cartpole example to demonstrate how to create an :class:`~assets.ArticulationCfg`.
The Cartpole is a simple robot that consists of a cart with a pole attached to it. The cart
is free to move along a rail, and the pole is free to rotate about the cart. The file for this configuration example is
``source/isaaclab_assets/isaaclab_assets/robots/cartpole.py``.

.. dropdown:: Code for Cartpole configuration
   :icon: code

   .. literalinclude:: ../../../source/isaaclab_assets/isaaclab_assets/robots/cartpole.py
      :language: python
      :linenos:


.. _asset-config-backends:

Choosing shared and backend-specific settings
---------------------------------------------

Use the shared :class:`~assets.ArticulationCfg` for the robot's initial state and actuators.
The simulation selects the physics backend separately.

For common USD properties, use classes from ``isaaclab.sim.schemas``, such as
:class:`~sim.schemas.RigidBodyBaseCfg` and :class:`~sim.schemas.ArticulationRootBaseCfg`.
Use ``isaaclab_physx.sim.schemas.Physx*Cfg`` for PhysX tuning and
``isaaclab_newton.sim.schemas.Newton*Cfg`` / ``Mujoco*Cfg`` for Newton and MJWarp-specific
settings. The :ref:`schema-cfgs` guide explains the available classes and their USD namespaces;
:doc:`../concepts/schema_fragments` shows how to author both backends' attributes
in one spawn configuration.

The Cartpole below uses the compatibility names ``RigidBodyPropertiesCfg`` and
``ArticulationRootPropertiesCfg``. Its PhysX solver iterations and sleep thresholds do not
configure Newton's solver. When adapting it to Newton, retain the shared initial-state and
actuator configuration and configure the Newton solver separately. For example, to override
Newton's self-collision setting:

.. code-block:: python

   from isaaclab_assets import CARTPOLE_CFG
   from isaaclab_newton.sim.schemas import NewtonArticulationRootPropertiesCfg

   robot_cfg = CARTPOLE_CFG.copy()
   robot_cfg.spawn.articulation_props = NewtonArticulationRootPropertiesCfg(self_collision_enabled=False)

See :doc:`../concepts/solver-tuning/tune_mjwarp` for solver settings,
:doc:`prepare_asset_for_newton`
for asset tuning, and :ref:`import-new-asset-multi-backend` for converted USD variants.


Defining the spawn configuration
--------------------------------

As explained in :ref:`tutorial-spawn-prims` tutorials, the spawn configuration defines
the properties of the assets to be spawned. This spawning may happen procedurally, or
through an existing asset file (e.g. USD or URDF). In this example, we will spawn the
Cartpole from a USD file.

When spawning an asset from a USD file, we define its :class:`~sim.spawners.from_files.UsdFileCfg`.
This configuration object takes in the following parameters:

* :class:`~sim.spawners.from_files.UsdFileCfg.usd_path`: The USD file path to spawn from
* :class:`~sim.spawners.from_files.UsdFileCfg.rigid_props`: The properties of the articulation's rigid-body links
* :class:`~sim.spawners.from_files.UsdFileCfg.articulation_props`: The properties of the articulation root

The last two parameters are optional. If not specified, they are kept at their default values in the USD file.

.. literalinclude:: ../../../source/isaaclab_assets/isaaclab_assets/robots/cartpole.py
   :language: python
   :start-at:     spawn=sim_utils.UsdFileCfg(
   :end-before:     init_state=
   :dedent:

To import articulation from a URDF file instead of a USD file, you can replace the
:class:`~sim.spawners.from_files.UsdFileCfg` with a :class:`~sim.spawners.from_files.UrdfFileCfg`.
For more details, please check the API documentation.


Defining the initial state
--------------------------

Every asset requires defining their initial or *default* state in the simulation through its configuration.
This configuration is stored into the asset's default state buffers that can be accessed when the asset's
state needs to be reset.

.. note::
   The initial state of an asset is defined w.r.t. its local environment frame. This then needs to
   be transformed into the global simulation frame when resetting the asset's state. For more
   details, please check the :ref:`tutorial-interact-articulation` tutorial.


For an articulation, the :class:`~assets.ArticulationCfg.InitialStateCfg` object defines the
initial state of the root of the articulation and the initial state of all its joints. In this
example, we will spawn the Cartpole at the origin of the XY plane at a Z height of 2.0 meters.
Meanwhile, the joint positions and velocities are set to 0.0.

.. literalinclude:: ../../../source/isaaclab_assets/isaaclab_assets/robots/cartpole.py
   :language: python
   :start-at:     init_state=
   :end-before:     actuators=
   :dedent:

Defining the actuator configuration
-----------------------------------

Actuators are a crucial component of an articulation. Through this configuration, it is possible
to define the type of actuator model to use. We can use the internal actuator model provided by
the physics engine (i.e. the implicit actuator model), or use a custom actuator model which is
governed by a user-defined system of equations (i.e. the explicit actuator model).
For more details on actuators, see :ref:`overview-actuators`.

The cartpole's articulation has two actuators, one corresponding to its each joint:
``cart_to_pole`` and ``slider_to_cart``. We use two different actuator models for these actuators as
an example. However, since they are both using the same actuator model, it is possible
to combine them into a single actuator model.

.. dropdown:: Actuator model configuration with separate actuator models
   :icon: code

   .. literalinclude:: ../../../source/isaaclab_assets/isaaclab_assets/robots/cartpole.py
      :language: python
      :start-at:     actuators=
      :end-at:     },
      :dedent:


.. dropdown:: Actuator model configuration with a single actuator model
   :icon: code

   .. code-block:: python

      actuators={
         "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            joint_effort_limit=400.0,
            joint_velocity_limit=100.0,
            stiffness={"slider_to_cart": 0.0, "cart_to_pole": 0.0},
            damping={"slider_to_cart": 10.0, "cart_to_pole": 0.0},
         ),
      },

.. note::
   Newton resolves the target mode of joints configured with
   :class:`~isaaclab.actuators.ImplicitActuatorCfg` before solver construction:
   stiffness-only selects position mode, damping-only velocity mode, both gains
   combined position/velocity mode, and zero gains effort mode. A gain of
   ``None`` retains the imported USD value; explicit actuator configurations use
   effort mode. Zero-gain USD drives therefore need no placeholder solely for a
   configured actuator. See :ref:`import-new-asset-ensure-drives-exist` for when
   :attr:`~isaaclab.sim.schemas.JointDrivePropertiesCfg.ensure_drives_exist`
   remains useful.


ActuatorCfg velocity/effort limits considerations
-------------------------------------------------

Use the following fields in an actuator configuration. They select joints and are resolved when the
articulation is constructed; the canonical runtime values live on
:class:`~isaaclab.assets.ArticulationData`. See :ref:`actuators-joint-property-ownership` for the
ownership model and runtime mutation paths.

.. list-table:: Limit configuration
    :header-rows: 1
    :widths: 28 36 36

    * - Field
      - Implicit actuator
      - Explicit actuator
    * - ``joint_effort_limit``
      - Writes the solver drive effort limit.
      - Writes the solver effort limit; defaults high to avoid a second model clip.
    * - ``actuator_effort_limit``
      - Not supported.
      - Clips actuator-model output.
    * - ``joint_velocity_limit``
      - Requests a solver velocity constraint.
      - Requests a solver velocity constraint.
    * - ``actuator_velocity_limit``
      - Creates the soft velocity-limit snapshot; it is not a solver request.
      - Describes the actuator rated speed; speed-dependent models use it in their torque curve.
    * - ``effort_limit``
      - Deprecated alias for ``joint_effort_limit``.
      - Deprecated alias for ``actuator_effort_limit``.
    * - ``velocity_limit``
      - Deprecated alias for ``actuator_velocity_limit``.
      - Deprecated alias for ``actuator_velocity_limit``.

Solver velocity enforcement is backend-dependent. ``joint_velocity_limit`` records the requested
joint state but is not a backend-independent safety clamp; see :ref:`newton-velocity-limits`.


USD vs. ActuatorCfg discrepancy resolution
------------------------------------------

USD having default value and the fact that ActuatorCfg can be specified with None, or a overriding value can sometime be
confusing what exactly gets written into simulation. The resolution follows these simple rules,per joint and per
property:

.. table:: Resolution Rules for USD vs. ActuatorCfg

    +------------------------+------------------------+--------------------+
    | **Condition**          | **ActuatorCfg Value**  | **Applied**        |
    +========================+========================+====================+
    | No override provided   | Not Specified          | USD Value          |
    +------------------------+------------------------+--------------------+
    | Override provided      | User's ActuatorCfg     | Same as ActuatorCfg|
    +------------------------+------------------------+--------------------+


Digging into USD can sometime be unconvinent, to help clarify what exact value is written, we designed a flag
:attr:`~isaaclab.assets.ArticulationCfg.actuator_value_resolution_debug_print`,
to help user figure out what exact value gets used in simulation.

Whenever an actuator parameter is overridden in the user's ActuatorCfg (or left unspecified),
we compare it to the value read from the USD definition and record any differences.  For each joint and each property,
if unmatching value is found, we log the resolution:

  1. **USD Value**
     The default limit or gain parsed from the USD asset.

  2. **ActuatorCfg Value**
     The user-provided override (or “Not Specified” if none was given).

  3. **Applied**
     The final value actually used for simulation: if the user didn't override it, this matches the USD value;
     otherwise it reflects the user's setting.

This resolution info is emitted as a warning table only when discrepancies exist.
Here's an example of what you'll see::

    +----------------+------------------------+---------------------+----+-------------+--------------------+----------+
    |     Group      |      Property          |         Name        | ID |  USD Value  | ActuatorCfg Value  | Applied  |
    +----------------+------------------------+---------------------+----+-------------+--------------------+----------+
    | panda_shoulder | joint_velocity_limit   |    panda_joint1     |  0 |    2.17e+00 |   Not Specified    | 2.17e+00 |
    |                |                        |    panda_joint2     |  1 |    2.17e+00 |   Not Specified    | 2.17e+00 |
    |                |                        |    panda_joint3     |  2 |    2.17e+00 |   Not Specified    | 2.17e+00 |
    |                |                        |    panda_joint4     |  3 |    2.17e+00 |   Not Specified    | 2.17e+00 |
    |                |     stiffness          |    panda_joint1     |  0 |    2.29e+04 |      8.00e+01      | 8.00e+01 |
    |                |                        |    panda_joint2     |  1 |    2.29e+04 |      8.00e+01      | 8.00e+01 |
    |                |                        |    panda_joint3     |  2 |    2.29e+04 |      8.00e+01      | 8.00e+01 |
    |                |                        |    panda_joint4     |  3 |    2.29e+04 |      8.00e+01      | 8.00e+01 |
    |                |      damping           |    panda_joint1     |  0 |    4.58e+03 |      4.00e+00      | 4.00e+00 |
    |                |                        |    panda_joint2     |  1 |    4.58e+03 |      4.00e+00      | 4.00e+00 |
    |                |                        |    panda_joint3     |  2 |    4.58e+03 |      4.00e+00      | 4.00e+00 |
    |                |                        |    panda_joint4     |  3 |    4.58e+03 |      4.00e+00      | 4.00e+00 |
    |                |      armature          |    panda_joint1     |  0 |    0.00e+00 |   Not Specified    | 0.00e+00 |
    |                |                        |    panda_joint2     |  1 |    0.00e+00 |   Not Specified    | 0.00e+00 |
    |                |                        |    panda_joint3     |  2 |    0.00e+00 |   Not Specified    | 0.00e+00 |
    |                |                        |    panda_joint4     |  3 |    0.00e+00 |   Not Specified    | 0.00e+00 |
    | panda_forearm  | joint_velocity_limit   |    panda_joint5     |  4 |    2.61e+00 |   Not Specified    | 2.61e+00 |
    |                |                        |    panda_joint6     |  5 |    2.61e+00 |   Not Specified    | 2.61e+00 |
    |                |                        |    panda_joint7     |  6 |    2.61e+00 |   Not Specified    | 2.61e+00 |
    |                |     stiffness          |    panda_joint5     |  4 |    2.29e+04 |      8.00e+01      | 8.00e+01 |
    |                |                        |    panda_joint6     |  5 |    2.29e+04 |      8.00e+01      | 8.00e+01 |
    |                |                        |    panda_joint7     |  6 |    2.29e+04 |      8.00e+01      | 8.00e+01 |
    |                |      damping           |    panda_joint5     |  4 |    4.58e+03 |      4.00e+00      | 4.00e+00 |
    |                |                        |    panda_joint6     |  5 |    4.58e+03 |      4.00e+00      | 4.00e+00 |
    |                |                        |    panda_joint7     |  6 |    4.58e+03 |      4.00e+00      | 4.00e+00 |
    |                |      armature          |    panda_joint5     |  4 |    0.00e+00 |   Not Specified    | 0.00e+00 |
    |                |                        |    panda_joint6     |  5 |    0.00e+00 |   Not Specified    | 0.00e+00 |
    |                |                        |    panda_joint7     |  6 |    0.00e+00 |   Not Specified    | 0.00e+00 |
    |                |      friction          |    panda_joint5     |  4 |    0.00e+00 |   Not Specified    | 0.00e+00 |
    |                |                        |    panda_joint6     |  5 |    0.00e+00 |   Not Specified    | 0.00e+00 |
    |                |                        |    panda_joint7     |  6 |    0.00e+00 |   Not Specified    | 0.00e+00 |
    |  panda_hand    | joint_velocity_limit   | panda_finger_joint1 |  7 |    2.00e-01 |   Not Specified    | 2.00e-01 |
    |                |                        | panda_finger_joint2 |  8 |    2.00e-01 |   Not Specified    | 2.00e-01 |
    |                |     stiffness          | panda_finger_joint1 |  7 |    1.00e+06 |      2.00e+03      | 2.00e+03 |
    |                |                        | panda_finger_joint2 |  8 |    1.00e+06 |      2.00e+03      | 2.00e+03 |
    |                |      armature          | panda_finger_joint1 |  7 |    0.00e+00 |   Not Specified    | 0.00e+00 |
    |                |                        | panda_finger_joint2 |  8 |    0.00e+00 |   Not Specified    | 0.00e+00 |
    |                |      friction          | panda_finger_joint1 |  7 |    0.00e+00 |   Not Specified    | 0.00e+00 |
    |                |                        | panda_finger_joint2 |  8 |    0.00e+00 |   Not Specified    | 0.00e+00 |
    +----------------+------------------------+---------------------+----+-------------+--------------------+----------+

To keep the cleaniness of logging, :attr:`~isaaclab.assets.ArticulationCfg.actuator_value_resolution_debug_print`
default to False, remember to turn it on when wishes.


.. _robot-configuration-example:

Example: configure and run two robots
-------------------------------------

The runnable example ``scripts/tutorials/01_assets/add_new_robot.py`` contrasts a minimal
Jetbot configuration with a more detailed Dofbot configuration. Start with an imported USD
asset (see :doc:`import_new_asset`) and define its spawn properties and actuators. Jetbot
retains the joint gains authored in the USD by setting stiffness and damping to ``None``:

.. literalinclude:: ../../../scripts/tutorials/01_assets/add_new_robot.py
   :language: python
   :start-at: JETBOT_CONFIG =
   :end-before: DOFBOT_CONFIG =

Dofbot additionally sets initial joint positions, groups joints by name, and specifies
actuator gains and limits. Its solver iterations and maximum depenetration velocity are
PhysX-specific; use :ref:`asset-config-backends` when adapting these properties to Newton.

.. dropdown:: Expanded Dofbot configuration from the runnable example
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/01_assets/add_new_robot.py
      :language: python
      :start-at: DOFBOT_CONFIG =
      :end-before: class NewRobotsSceneCfg

The example adds both configurations to an ``InteractiveSceneCfg``, assigns each robot a
path under every environment, and constructs the scene. Its loop resets root and joint
states, sets joint targets, writes commands, steps physics, and updates the scene buffers.
See :ref:`tutorial-interactive-scene` for scene construction and
:ref:`tutorial-interact-articulation` for the reset and control loop.

.. dropdown:: Complete runnable example
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/01_assets/add_new_robot.py
      :language: python
      :linenos:

Run the example in the Isaac Sim viewport:

.. code-block:: bash

   uv run isaaclab -p scripts/tutorials/01_assets/add_new_robot.py --viz kit

This example uses PhysX physics and requires Isaac Sim. The Dofbot gripper is not actuated
in this example, so a warning about unconfigured joints is expected. Stop the example with ``Ctrl+C``.
