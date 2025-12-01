Simulation Performance  and Tuning
====================================

The performance of the simulation can be affected by various factors, including the number of objects in the scene,
the complexity of the physics simulation, and the hardware being used. Here are some tips to improve performance:

1. **Use Headless Mode**: Running the simulation in headless mode can significantly improve performance, especially
   when rendering is not required. You can enable headless mode by using the ``--headless`` flag when running the
   simulator.
2. **Avoid Unnecessary Collisions**: If possible, reduce the number of object overlaps to reduce overhead in the simulation.
   Excessive contacts and collisions in the simulation can be expensive in the collision phase in the simulation.
3. **Use Simplified Physics**: Consider using simplified physics collision geometries or lowering simulation fidelity
   for better performance. This can be done by modifying the assets and adjusting the physics parameters in the simulation configuration.
4. **Use CPU/GPU Simulation**: If your scene consists of just a few articulations or rigid bodies, consider using CPU simulation
   for better performance. For larger scenes, using GPU simulation can significantly improve performance.

Collision Geometries
--------------------

Collision geometries are used to define the shape of objects in the simulation for collision detection. Using
simplified collision geometries can improve performance and reduce the complexity of the simulation.

For example, if you have a complex mesh, you can create a simplified collision geometry that approximates the shape
of the mesh. This can be done in Isaac Sim through the UI by modifying the collision mesh and approximation methods.

Additionally, we can often remove collision geometries on areas of the robot that are not important for training.
In the Anymal-C robot, we keep the collision geometries for the kneeds and feet, but remove the collision geometries
on other parts of the legs to optimize for performance.

Simpler collision geometries such as primitive shapes like spheres will also yield better performance than complex meshes.
For example, an SDF mesh collider will be more expensive than a simple sphere.

Note that cylinder and cone collision geometries have special support for smooth collisions with triangle meshes for
better wheeled simulation behavior. This comes at a cost of performance and may not always be desired. To disable this feature,
we can set the stage settings ``--/physics/collisionApproximateCylinders=true`` and ``--/physics/collisionApproximateCones=true``.

Another item to watch out for in GPU RL workloads is warnings about GPU compatibility of ``Convex Hull`` approximated mesh collision geometry.
If the input mesh has a high aspect ratio (e.g. a long thin shape), the convex hull approximation may be incompatible with GPU simulation,
triggering a CPU fallback that can significantly impact performance.

A CPU-fallback warning looks as follows: ``[Warning] [omni.physx.cooking.plugin] ConvexMeshCookingTask: failed to cook GPU-compatible mesh,
collision detection will fall back to CPU. Collisions with particles and deformables will not work with this mesh.``.
Suitable workarounds include switching to a bounding cube approximation, or using a static triangle mesh collider
if the geometry is not part of a dynamic rigid body.

CPU Governor Settings on Linux
------------------------------

CPU governors dictate the operating clock frequency range and scaling of the CPU. This can be a limiting factor for Isaac Sim performance. For maximum performance, the CPU governor should be set to ``performance``. To modify the CPU governor, run the following commands:

.. code-block:: bash

    sudo apt-get install linux-tools-common
    cpupower frequency-info # Check available governors
    sudo cpupower frequency-set -g performance # Set governor with root permissions

.. note::

    Not all governors are available on all systems. Governors enabling higher clock speed are typically more performance-centric and will yield better performance for Isaac Sim.

Additional Performance Guides
-----------------------------

There are many ways to "tune" the performance of the simulation, but the way you choose largely depends on what you are trying to simulate. In general, the first place
you will want to look for performance gains is with the `physics engine <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides.html>`_. Next to rendering
and running deep learning models, the physics engine is the most computationally costly. Tuning the physics sim to limit the scope to only the task of interest is a great place to
start hunting for performance gains.

We have recently released a new `gripper tuning guide <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/gripper_tuning_example.html>`_ , specific to contact and grasp tuning. Please check it first if you intend to use robot grippers. For additional details, you should also checkout these guides!

* `Isaac Sim Performance Optimization Handbook <https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html>`_
* `Omni Physics Simulation Performance Guide <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/physics-performance.html>`_
