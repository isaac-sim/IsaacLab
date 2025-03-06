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
is free to move along a rail, and the pole is free to rotate about the cart.

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
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness={"slider_to_cart": 0.0, "cart_to_pole": 0.0},
            damping={"slider_to_cart": 10.0, "cart_to_pole": 0.0},
         ),
      },
