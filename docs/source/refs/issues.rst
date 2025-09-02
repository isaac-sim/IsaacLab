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

.. note::
    With Isaac Lab 1.2, we have introduced a PhysX kinematic update call inside the
    :attr:`~isaaclab.assets.ArticulationData.body_state_w` attribute. This workaround
    ensures that the states of the links are updated when the root state or joint state
    of an articulation is set.


Blank initial frames from the camera
------------------------------------

When using the :class:`~isaaclab.sensors.Camera` sensor in standalone scripts, the first few frames
may be blank. This is a known issue with the simulator where it needs a few steps to load the material
textures properly and fill up the render targets.

A hack to work around this is to add the following after initializing the camera sensor and setting
its pose:

.. code-block:: python

    from isaaclab.sim import SimulationContext

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
.. _Omniverse Isaac Sim documentation: https://docs.isaacsim.omniverse.nvidia.com/latest/overview/known_issues.html#


Exiting the process
-------------------

When exiting a process with ``Ctrl+C``, occasionally the below error may appear:

.. code-block:: bash

	[Error] [omni.physx.plugin] Subscription cannot be changed during the event call.

This is due to the termination occurring in the middle of a physics event call and
should not affect the functionality of Isaac Lab. It is safe to ignore the error
message and continue with terminating the process. On Windows systems, please use
``Ctrl+Break`` or ``Ctrl+fn+B`` to terminate the process.


GLIBCXX errors in Conda
-----------------------

In Isaac Sim 5.0, we have observed some workflows exiting with an ``OSError`` indicating
``version 'GLIBCXX_3.4.30' not found`` when running from a conda environment.
The issue apperas to be stemming from importing torch or torch-related packages, such as tensorboard,
prior to launching ``AppLauncher``. As a workaround, ensure that all torch imports happen after
the ``AppLauncher`` instance has been created, which should resolve the error.
