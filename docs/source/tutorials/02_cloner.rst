Duplicating a scene efficiently
===============================

In the previous tutorial, we needed to spawn each individual prim manually to create multiple robots in the scene.
This operation can be cumbersome and slow when defining a large scene which needs to duplicated a large number of
times, such for reinforement learning. In this tutorial we will look at duplicating the scene with the cloner APIs
provided by Isaac Sim.

.. note::
    A more descriptive tutorial on the cloner APIs can be found in the `cloner API tutorial
    <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_gym_cloner.html>`_. We introduce its main
    concepts in this tutorial for convenience.

The Code
~~~~~~~~

The tutorial corresponds to the ``play_cloner.py`` script in the ``orbit/source/standalone/demo`` directory.


.. literalinclude:: ../../../source/standalone/demo/play_cloner.py
   :language: python
   :emphasize-lines: 37,61-67,69-73,94,105,107-120,126
   :linenos:

The Code Explained
~~~~~~~~~~~~~~~~~~

Configuring the simulation stage
--------------------------------

Cloning a scene to a large number of instances can lead to slow accessibility of the physics buffers from Python. This
is because in Omniverse, all read and write happens through the intermediate USD layer. Thus, whenever a physics
step happens. the data is first written into the USD buffers from the physics tensor buffers, and then available
to the users. Thus, to avoid this overhead, we can configure the simulation stage to write the data directly into
the tensors accessed by the users. This is performed by enabling the
`PhysX flatcache <https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_physics.html#physx-short-flatcache-also-known-as-fabric-rename-in-next-release>`_.

However, once flatcache is enabled, the rendering is not updated since it still reads from the USD buffers, which are
no longer being updated. Thus, we also need to enable scene graph instancing to render the scene.

These are done by setting the following flags:

.. literalinclude:: ../../../source/standalone/demo/play_cloner.py
   :language: python
   :lines: 61-67
   :linenos:
   :lineno-start: 61


Cloning the scene
-----------------

The basic :class:`Cloner` class clones all the prims under a source prim path and puts them under the specified paths.
It provides a method to generate the paths for the cloned prims based on the number of clones. The generated paths
follow the pattern of the source prim path with a number appended to the end. For example, if the source prim path is
`/World/envs/env_0`, the generated paths will be `/World/envs/env_0`, `/World/envs/env_1`, `/World/envs/env_2`, etc.
These are then passed to the :meth:`clone()` method to clone the scene.

In this tutorial, we use the :class:`GridCloner` class which clones the scene in a grid pattern. The grid spacing
defines the distance between the cloned scenes. We define the base environment path to be `/World/envs`, which is
the parent prim of all the cloned scenes. The source prim path is then defined to be `/World/envs/env_0`. All the
prims under this prim path are cloned.

.. literalinclude:: ../../../source/standalone/demo/play_cloner.py
   :language: python
   :lines: 69-73
   :linenos:
   :lineno-start: 69

Unlike the previous tutorial, in this tutorial, we only spawn one environment and clone it to multiple instances.
We spawn the table and the robot under the source prim path.

.. literalinclude:: ../../../source/standalone/demo/play_cloner.py
   :language: python
   :lines: 92-105
   :linenos:
   :lineno-start: 92

The :meth:`generate_paths()` method generates the paths for the cloned prims. The generated paths are then passed to
the :meth:`clone()` method to clone the source scene. It returns the positions of the cloned scenes relative to
the simulation world origin.

.. literalinclude:: ../../../source/standalone/demo/play_cloner.py
   :language: python
   :lines: 107-120
   :linenos:
   :lineno-start: 107


Applying collision filtering
----------------------------

Collisions between the cloned environments is filtered by using the :meth:`filter_collisions()` method. This
is done by specifying the physics scene path, the collision prim path, the cloned prim paths, and the global prim paths.
The global prim paths are the prims that are not cloned and are shared between the cloned scenes. For instance,
the ground plane belongs to the global prim paths.

.. literalinclude:: ../../../source/standalone/demo/play_cloner.py
   :language: python
   :lines: 116-120
   :linenos:
   :lineno-start: 116

The Code Execution
~~~~~~~~~~~~~~~~~~

Now that we have gone through the code, let's run the script and see the result:

.. code-block:: bash

   ./orbit.sh -p source/standalone/demo/play_cloner.py --robot franka_panda --num_robots 128

This should behave the same as the previous tutorial, except that it is much faster to spawn the robots.
To stop the simulation, you can either close the window, or press the ``STOP`` button in the UI, or
press ``Ctrl+C`` in the terminal.

Now that we have learned how to design a scene with a robot and clone it multiple times, we can move on to
the next tutorial to learn how to control the robot with a task-space controller.
