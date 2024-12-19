.. _tutorial-interact-articulation:

Interacting with an articulation
================================

.. currentmodule:: isaaclab


This tutorial shows how to interact with an articulated robot in the simulation. It is a continuation of the
:ref:`tutorial-interact-rigid-object` tutorial, where we learned how to interact with a rigid object.
On top of setting the root state, we will see how to set the joint state and apply commands to the articulated
robot.


The Code
~~~~~~~~

The tutorial corresponds to the ``run_articulation.py`` script in the ``scripts/tutorials/01_assets``
directory.

.. dropdown:: Code for run_articulation.py
   :icon: code

   .. literalinclude:: ../../../../scripts/tutorials/01_assets/run_articulation.py
      :language: python
      :emphasize-lines: 58-69, 91-104, 108-111, 116-117
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

Designing the scene
-------------------

Similar to the previous tutorial, we populate the scene with a ground plane and a distant light. Instead of
spawning rigid objects, we now spawn a cart-pole articulation from its USD file. The cart-pole is a simple robot
consisting of a cart and a pole attached to it. The cart is free to move along the x-axis, and the pole is free to
rotate about the cart. The USD file for the cart-pole contains the robot's geometry, joints, and other physical
properties.

For the cart-pole, we use its pre-defined configuration object, which is an instance of the
:class:`assets.ArticulationCfg` class. This class contains information about the articulation's spawning strategy,
default initial state, actuator models for different joints, and other meta-information. A deeper-dive into how to
create this configuration object is provided in the :ref:`how-to-write-articulation-config` tutorial.

As seen in the previous tutorial, we can spawn the articulation into the scene in a similar fashion by creating
an instance of the :class:`assets.Articulation` class by passing the configuration object to its constructor.

.. literalinclude:: ../../../../scripts/tutorials/01_assets/run_articulation.py
   :language: python
   :start-at: # Create separate groups called "Origin1", "Origin2"
   :end-at: cartpole = Articulation(cfg=cartpole_cfg)


Running the simulation loop
---------------------------

Continuing from the previous tutorial, we reset the simulation at regular intervals, set commands to the articulation,
step the simulation, and update the articulation's internal buffers.

Resetting the simulation
""""""""""""""""""""""""

Similar to a rigid object, an articulation also has a root state. This state corresponds to the root body in the
articulation tree. On top of the root state, an articulation also has joint states. These states correspond to the
joint positions and velocities.

To reset the articulation, we first set the root state by calling the :meth:`Articulation.write_root_pose_to_sim` and :meth:`Articulation.write_root_velocity_to_sim`
methods. Similarly, we set the joint states by calling the :meth:`Articulation.write_joint_state_to_sim` method.
Finally, we call the :meth:`Articulation.reset` method to reset any internal buffers and caches.

.. literalinclude:: ../../../../scripts/tutorials/01_assets/run_articulation.py
   :language: python
   :start-at: # reset the scene entities
   :end-at: robot.reset()

Stepping the simulation
"""""""""""""""""""""""

Applying commands to the articulation involves two steps:

1. *Setting the joint targets*: This sets the desired joint position, velocity, or effort targets for the articulation.
2. *Writing the data to the simulation*: Based on the articulation's configuration, this step handles any
   :ref:`actuation conversions <overview-actuators>` and writes the converted values to the PhysX buffer.

In this tutorial, we control the articulation using joint effort commands. For this to work, we need to set the
articulation's stiffness and damping parameters to zero. This is done a-priori inside the cart-pole's pre-defined
configuration object.

At every step, we randomly sample joint efforts and set them to the articulation by calling the
:meth:`Articulation.set_joint_effort_target` method. After setting the targets, we call the
:meth:`Articulation.write_data_to_sim` method to write the data to the PhysX buffer. Finally, we step
the simulation.

.. literalinclude:: ../../../../scripts/tutorials/01_assets/run_articulation.py
   :language: python
   :start-at: # Apply random action
   :end-at: robot.write_data_to_sim()


Updating the state
""""""""""""""""""

Every articulation class contains a :class:`assets.ArticulationData` object. This stores the state of the
articulation. To update the state inside the buffer, we call the :meth:`assets.Articulation.update` method.

.. literalinclude:: ../../../../scripts/tutorials/01_assets/run_articulation.py
   :language: python
   :start-at: # Update buffers
   :end-at: robot.update(sim_dt)


The Code Execution
~~~~~~~~~~~~~~~~~~


To run the code and see the results, let's run the script from the terminal:

.. code-block:: bash

   ./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py


This command should open a stage with a ground plane, lights, and two cart-poles that are moving around randomly.
To stop the simulation, you can either close the window, or press ``Ctrl+C`` in the terminal.

.. figure:: ../../_static/tutorials/tutorial_run_articulation.jpg
    :align: center
    :figwidth: 100%
    :alt: result of run_articulation.py

In this tutorial, we learned how to create and interact with a simple articulation. We saw how to set the state
of an articulation (its root and joint state) and how to apply commands to it. We also saw how to update its
buffers to read the latest state from the simulation.

In addition to this tutorial, we also provide a few other scripts that spawn different robots.These are included
in the ``scripts/demos`` directory. You can run these scripts as:

.. code-block:: bash

   # Spawn many different single-arm manipulators
   ./isaaclab.sh -p scripts/demos/arms.py

   # Spawn many different quadrupeds
   ./isaaclab.sh -p scripts/demos/quadrupeds.py
