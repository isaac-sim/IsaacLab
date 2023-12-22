Known issues
============

Blank initial frames from the camera
------------------------------------

When using the :class:`Camera` sensor in standalone scripts, the first few frames may be blank.
This is a known issue with the simulator where it needs a few steps to load the material
textures properly and fill up the render targets.

A hack to work around this is to add the following after initializing the camera sensor and setting
its pose:

.. code-block:: python

    from omni.isaac.core.simulation_context import SimulationContext

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
