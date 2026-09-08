.. _how-to-write-articulation-config:


Writing an Asset Configuration
==============================

.. currentmodule:: isaaclab

This guide walks through the process of creating an :class:`~assets.ArticulationCfg`.
The :class:`~assets.ArticulationCfg` is a configuration object that defines the
properties of an :class:`~assets.Articulation` in Isaac Lab.

.. note::

   While we only cover the creation of an :class:`~assets.ArticulationCfg` in this guide,
   the process is similar for creating any other asset configuration object.

We will use the Cartpole example to demonstrate how to create an :class:`~assets.ArticulationCfg`.
The Cartpole is a simple robot that consists of a cart with a pole attached to it. The cart
is free to move along a rail, and the pole is free to rotate about the cart. The file for this configuration example is
``source/isaaclab_assets/isaaclab_assets/robots/cartpole.py``.

.. dropdown:: Code for Cartpole configuration
   :icon: code

   .. literalinclude:: ../../../source/isaaclab_assets/isaaclab_assets/robots/cartpole.py
      :language: python
      :linenos:


Defining the spawn configuration
--------------------------------

As explained in :ref:`tutorial-spawn-prims` tutorials, the spawn configuration defines
the properties of the assets to be spawned. This spawning may happen procedurally, or
through an existing asset file (e.g. USD or URDF). In this example, we will spawn the
Cartpole from a USD file.

When spawning an asset from a USD file, we define its :class:`~sim.spawners.from_files.UsdFileCfg`.
This configuration object takes in the following parameters:

* :class:`~sim.spawners.from_files.UsdFileCfg.usd_path`: The USD file path to spawn from
* :class:`~sim.spawners.from_files.UsdFileCfg.rigid_props`: The properties of the articulation's root
* :class:`~sim.spawners.from_files.UsdFileCfg.articulation_props`: The properties of all the articulation's links

The last two parameters are optional. If not specified, they are kept at their default values in the USD file.

.. literalinclude:: ../../../source/isaaclab_assets/isaaclab_assets/robots/cartpole.py
   :language: python
   :lines: 19-35
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
   :lines: 36-38
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
      :lines: 39-49
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
