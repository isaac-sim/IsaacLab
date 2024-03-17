Known Issues
============

.. attention::

    Please also refer to the `Omniverse Isaac Sim documentation`_ for known issues and workarounds.

Stale values after resetting the environment
--------------------------------------------

When resetting the environment, some of the data fields of assets and sensors are not updated.
These include the poses of links in a kinematic chain, the camera images, the contact sensor readings,
and the lidar point clouds. This is a known issue which has to do with the way the PhysX and
rendering engines work in Omniverse.

Many physics engines do a simulation step as a two-level call: ``forward()`` and ``simulate()``,
where the kinematic and dynamic states are updated, respectively. Unfortunately, PhysX has only a
single ``step()`` call where the two operations are combined. Due to computations through GPU
kernels, it is not so straightforward for them to split these operations. Thus, at the moment,
it is not possible to set the root and/or joint states and do a forward call to update the
kinematic states of links. This affects both initialization as well as episodic resets.

Similarly for RTX rendering related sensors (such as cameras), the sensor data is not updated
immediately after setting the state of the sensor. The rendering engine update is bundled with
the simulator's ``step()`` call which only gets called when the simulation is stepped forward.
This means that the sensor data is not updated immediately after a reset and it will hold
outdated values.

While the above is erroneous, there is currently no direct workaround for it. From our experience in
using IsaacGym, the reset values affect the agent learning critically depending on how frequently
the environment terminates. Eventually if the agent is learning successfully, this number drops
and does not affect the performance that critically.

We have made a feature request to the respective Omniverse teams to have complete control
over stepping different parts of the simulation app. However, at this point, there is no set
timeline for this feature request.


Non-determinism in physics simulation
-------------------------------------

Due to GPU work scheduling, there's a possibility that runtime changes to simulation parameters
may alter the order in which operations take place. This occurs because environment updates can
happen while the GPU is occupied with other tasks. Due to the inherent nature of floating-point
numeric storage, any modification to the execution ordering can result in minor changes in the
least significant bits of output data. These changes may lead to divergent execution over the
course of simulating thousands of environments and simulation frames.

An illustrative example of this issue is observed with the runtime domain randomization of object's
physics materials. This process can introduce both determinancy and simulation issues when executed
on the GPU due to the way these parameters are passed from the CPU to the GPU in the lower-level APIs.
Consequently, it is strongly advised to perform this operation only at setup time, before the
environment stepping commences.

For more information, please refer to the `PhysX Determinism documentation`_.


Blank initial frames from the camera
------------------------------------

When using the :class:`omni.isaac.orbit.sensors.Camera` sensor in standalone scripts, the first few frames
may be blank. This is a known issue with the simulator where it needs a few steps to load the material
textures properly and fill up the render targets.

A hack to work around this is to add the following after initializing the camera sensor and setting
its pose:

.. code-block:: python

    from omni.isaac.orbit.sim import SimulationContext

    sim = SimulationContext.instance()

    # note: the number of steps might vary depending on how complicated the scene is.
    for _ in range(12):
        sim.render()


Using instanceable assets for markers
-------------------------------------

When using `instanceable assets`_ for markers, the markers do not work properly, since Omniverse does not support
instanceable assets when using the :class:`UsdGeom.PointInstancer` schema. This is a known issue and will hopefully
be fixed in a future release.

If you use an instanceable assets for markers, the marker class removes all the physics properties of the asset.
This is then replicated across other references of the same asset since physics properties of instanceable assets
are stored in the instanceable asset's USD file and not in its stage reference's USD file.

.. _instanceable assets: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_gym_instanceable_assets.html
.. _Omniverse Isaac Sim documentation: https://docs.omniverse.nvidia.com/isaacsim/latest/known_issues.html
.. _PhysX Determinism documentation: https://nvidia-omniverse.github.io/PhysX/physx/5.3.1/docs/BestPractices.html#determinism
