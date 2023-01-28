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
