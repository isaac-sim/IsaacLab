Known issues
============

Installation errors due to gym==0.21.0
--------------------------------------

When installing the gym package, you may encounter the following error:

.. code-block::

    error in gym setup command: 'extras_require' must be a dictionary whose values are strings or lists of
    strings containing valid project/version requirement specifiers.
    ----------------------------------------
    ERROR: Could not find a version that satisfies the requirement gym==0.21.0 (from omni-isaac-orbit-envs[all])
    (from versions: 0.0.2, 0.0.3, 0.0.4, 0.0.5, 0.0.6, 0.0.7, 0.1.0, 0.1.1, 0.1.2, 0.1.3, 0.1.4, 0.1.5, 0.1.6,
    ...
    0.15.7, 0.16.0, 0.17.0, 0.17.1, 0.17.2, 0.17.3, 0.18.0, 0.18.3, 0.19.0, 0.20.0, 0.21.0, 0.22.0, 0.23.0,
    0.23.1, 0.24.0, 0.24.1, 0.25.0, 0.25.1, 0.25.2, 0.26.0, 0.26.1, 0.26.2)
    ERROR: No matching distribution found for gym==0.21.0


This issue arises since the ``setuptools`` package from version 67.0 onwards does not support malformed version strings.
Since the OpenAI Gym package that is no longer being maintained (`issue link <https://github.com/openai/gym/issues/3200>`_),
the current workaround is to install the ``setuptools`` package version 66.0.0. You can do this by running the following
command:

.. code-block:: bash

    ./orbit.sh -p -m pip install -U setuptools==66

Regression in Isaac Sim 2022.2.1
--------------------------------

In Isaac Sim 2022.2.1, we have noticed the following regression and issues that should be fixed in the
next release:

* The RTX-Lidar sensor does not work properly and returns empty data.
* The :class:`ArticulationView` class leads to issues when using GPU-physics pipeline.
* The :class:`UrdfImporter` does not load the off-diagonal elements of the inertia matrix properly. This
  leads to incorrect physics simulation of the robot.

Due to this regression, we recommend using Isaac Sim 2022.2.0 for now. We will update this section once
the issues are fixed in a future release.


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
