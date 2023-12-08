.. _interactive-scene:

Using InteractiveScene
======================

In previous tutorials, we've used :meth:`spawn` to manually spawn assets,
but in this tutorial we introduce :class:`InteractiveScene` and its
associated configuration class :class:`InteractiveSceneCfg`.

:class:`InteractiveScene` provides a few benefits over using assets'
:meth:`spawn` methods directly:

* collects all of the assets in a single configuration object which makes them easier
  to manage
* enables user-friendly cloning of scene elements for multiple environments
* user doesn't need to call :meth:`spawn` for each asset - this is handled implicitly

We will implement an :class:`InteractiveSceneCfg` to design a simple scene
for the Cartpole, which consists of a ground
plane, light, and the cartpole :class:`Articulation`.

The Code
~~~~~~~~
This tutorial corresponds to the ``scene_creation.py`` script within
``orbit/source/standalone/tutorials``.

.. dropdown:: :fa:`eye,mr-1` Code for ``scene_creation.py``

   .. literalinclude:: ../../../../source/standalone/tutorials/03_scene/scene_creation.py
      :language: python
      :linenos:

The Code Explained
~~~~~~~~~~~~~~~~~~

We compose our scene configuration by sub-classing :class:`InteractiveSceneCfg`. We
then add the elements we want as attributes of the class, which is a common pattern
for configuration classes in Orbit. In this case, we add a
ground plane, light, and cartpole. The names of the attributes (``ground``,
``robot``, ``dome_light``, ``distant_light``) are used as keys to access the
corresponding assets in the :class:`InteractiveScene` object.

.. literalinclude:: ../../../../source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/classic/cartpole/cartpole_scene.py
   :language: python
   :linenos:

Within the config elements of :class:`CartpoleSceneCfg`, notice that we pass
in a few arguments. We will describe these briefly as they are fundamental to
how the scene interface works - refer to in-code documentation for more depth.

* **prim_path**: The USD layer to associate the asset with
* **spawn**: The configuration object for spawning
* **init_state**: The desired initial pose of the asset. Defaults to identity if unspecified

You can find more about additional arguments by looking directly at the docstrings for
:class:`AssetBaseCfg`, :class:`RigidObjectCfg` and :class:`ArticulationCfg`.

:class:`InteractiveSceneCfg` is simply a collection of
asset configurations, so add new elements as you see fit
(including sensors, markers, terrain, etc.) you've learned to utilize in previous tutorials.

Setup Simulator and Spawn Scene
-------------------------------

.. literalinclude:: ../../../../source/standalone/tutorials/03_scene/scene_creation.py
   :language: python
   :start-after: # Main
   :end-before: # Extract cartpole from InteractiveScene

Accessing Scene Elements
------------------------

Individual scene elements can then be accessed from the :class:`InteractiveScene` via
the different asset groups: ``articulations``, ``rigid_objects``, ``sensors``,
``extras`` (where lights are found for instance). Each of these is a dictionary with the keys being assigned based
on the object instance name.

In the example script we access our cartpole :class:`Articulation` here using the ``"robot"`` key:

.. literalinclude:: ../../../../source/standalone/tutorials/03_scene/scene_creation.py
   :language: python
   :start-after: # Extract cartpole from InteractiveScene
   :end-before: # Simulation loop

Simulation Loop
---------------

.. literalinclude:: ../../../../source/standalone/tutorials/03_scene/scene_creation.py
   :language: python
   :emphasize-lines: 15-16, 18, 28
   :start-after: # Simulation loop
   :end-before: # End simulation loop

The rest of the script should look familiar to previous scripts that interfaced with :class:`Articulation`,
with a few small differences:

*  :meth:`Articulation.set_joint_position_target` and  :meth:`Articulation.set_joint_velocity_target`
   in combination with :meth:`InteractiveScene.write_data_to_sim`
   are used instead of :meth:`Articulation.write_joint_data_to_sim` to set the desired position and velocity targets
   without writing them to the simulation.
*  :meth:`InteractiveScene.update` is used in place of :meth:`Articulation.update`

Under the hood, ``InteractiveScene`` calls the ``update`` and ``write_data_to_sim`` for each asset in the scene,
so you only need to call these once per simulation step.

Cloning
-------

As mentioned previously, one of the key benefits of using :class:`InteractiveScene`
is its ability to handle cloning of assets seamlessly with the only user input
being the ``num_envs`` (The number of desired environments to spawn). The spacing between
environments is also configurable via ``env_spacing``.

We will exercise this below by passing in ``--num_envs 32`` to the tutorial script to spawn 32 cartpoles.

The Code Execution
~~~~~~~~~~~~~~~~~~

Now that we have gone through the code, let's run the script and see the result:

.. code-block:: bash

   ./orbit.sh -p source/standalone/tutorials/03_scene/scene_creation.py --num_envs 32


This should open a stage with 32 cartpoles. The simulation should be
playing with the poles of each cartpole balancing vertically.

In this tutorial we saw how to use :class:`InteractiveScene` to create a
scene with multiple assets. We also saw how to use the ``num_envs`` argument
to clone the scene for multiple environments.

.. note::
  There are many more examples of other ``InteractiveSceneCfg`` in the tasks found in
  ``source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks`` for
  reference.
