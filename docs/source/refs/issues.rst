Known Issues
============

.. attention::

    Please also refer to the `Omniverse Isaac Sim documentation`_ for known issues and workarounds.

Each entry below names the backends it affects. An issue listed under one backend does not
apply to the others unless it says so.

.. contents::
   :local:
   :depth: 2


PhysX backends
--------------

Surface grippers require CPU simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Affects:** ``physics=isaacsim_physx`` surface-gripper tasks.

Surface grippers require CPU simulation. This includes the UR10 Long/Short Suction stacking tasks,
the Galbot Right Arm Suction stacking task, and its relative and absolute Mimic variants.
Pass ``--device cpu`` when running teleoperation. Zero and random agents preserve these tasks'
CPU defaults when ``--device`` is omitted; an explicit GPU override is unsupported.

Sensor readings are stale immediately after a reset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Affects:** ``physics=isaacsim_physx``, ``physics=ovphysx``, and any RTX-based renderer.

Many physics engines do a simulation step as a two-level call: ``forward()`` and ``simulate()``,
where the kinematic and dynamic states are updated respectively. PhysX has only a single
``step()`` call where the two operations are combined. Because of computations through GPU
kernels, it is not straightforward to split them. As a result, writing a root or joint state
does not by itself run a full forward pass.

For **articulation link poses** this is handled: reading
:attr:`~isaaclab.assets.ArticulationData.body_link_pose_w` (or the deprecated
:attr:`~isaaclab.assets.ArticulationData.body_state_w`) triggers a PhysX kinematic update, so
link poses reflect a preceding root-state or joint-state write without an intervening
``step()``.

For **RTX rendering-based sensors** — cameras in particular — the data is still not refreshed
by a state write. The rendering engine update is bundled with the simulator's ``step()`` call,
so the sensor data is only refreshed when the simulation is stepped forward, and a read taken
between a reset and the next step returns the previous frame.

There is currently no direct workaround for the sensor case. From our experience, the reset
values affect agent learning in proportion to how frequently the environment terminates; as an
agent learns successfully, the termination rate drops and the effect becomes less significant.

Exiting the process
~~~~~~~~~~~~~~~~~~~

**Affects:** ``physics=isaacsim_physx``.

When exiting a process with ``Ctrl+C``, occasionally the below error may appear:

.. code-block:: bash

	[Error] [omni.physx.plugin] Subscription cannot be changed during the event call.

This is due to the termination occurring in the middle of a physics event call and
should not affect the functionality of Isaac Lab. It is safe to ignore the error
message and continue with terminating the process. On Windows systems, please use
``Ctrl+Break`` or ``Ctrl+fn+B`` to terminate the process.


Newton backends
---------------

OpenUSD can crash while parsing collider-rich rigid bodies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Affects:** Kitless Newton workflows using OpenUSD releases earlier than 26.05, and Isaac Sim
releases earlier than 6.1. Isaac Sim 6.1 and later contain the upstream fix in their bundled
OpenUSD runtime.

Older OpenUSD releases have a thread-safety issue in
``UsdPhysics.LoadUsdPhysicsFromRange`` when multiple colliders belong to the same rigid body.
Newton uses this API while importing a USD stage, so affected processes can terminate during
scene creation with a segmentation fault, access violation, heap-corruption message, or hang.
The issue and upstream fix are described in `OpenUSD PR #4002`_.

.. note::

   For kitless workflows, Isaac Lab obtains the ``pxr`` modules from ``usd-exchange``.
   ``usd-exchange`` and ``usd-core`` are alternative Python distributions of the same OpenUSD
   runtime and should not be installed together because both provide ``pxr``. They use different
   distribution version schemes: for example, ``usd-exchange==2.3.0`` provides OpenUSD 25.05.
   Isaac Sim instead uses its own Kit-bundled OpenUSD runtime.

As a workaround for an affected kitless or pre-6.1 Isaac Sim runtime, limit OpenUSD to one worker
thread. Set ``PXR_WORK_THREAD_LIMIT`` before launching Python so the limit is present before any
``pxr`` module initializes:

.. tab-set::

   .. tab-item:: Linux

      .. code-block:: bash

         PXR_WORK_THREAD_LIMIT=1 uv run isaaclab train --rl_library rsl_rl --task Isaac-Reach-Franka physics=newton_mjwarp

   .. tab-item:: Windows

      .. code-block:: batch

         set PXR_WORK_THREAD_LIMIT=1
         uv run isaaclab train --rl_library rsl_rl --task Isaac-Reach-Franka physics=newton_mjwarp

This environment variable limits OpenUSD's process-wide worker pool to one thread and can reduce
USD import performance. Use it only with affected runtimes, and remove it after upgrading to an
OpenUSD provider that contains the fix or to Isaac Sim 6.1 or later.

.. _known-issues-closed-loop-newton:

Closed-loop articulations are not validated on Kamino (e.g. Agility Digit)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Affects:** ``physics=newton_kamino``.

Robots whose USD encodes a closed kinematic loop — such as the achilles rod and toe push-rods
on the Agility Digit — are not validated on ``newton_kamino``.


Renderers
---------

Blank initial frames from the camera
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Affects:** RTX-based renderers (``renderer=isaacsim_rtx``, ``renderer=ovrtx``).

When using the :class:`~isaaclab.sensors.Camera` sensor in standalone scripts, the first few frames
may be blank. This is a known issue with the simulator where it needs a few steps to load the material
textures properly and fill up the render targets. It is most likely on a cold asset cache and in
scenes with many or large textures; simple scenes with locally cached assets often render content on
the very first frame.

If you do see blank frames, add the following after initializing the camera sensor and setting
its pose:

.. code-block:: python

    from isaaclab.sim import SimulationContext

    sim = SimulationContext.instance()

    # note: the number of steps might vary depending on how complicated the scene is.
    for _ in range(12):
        sim.render()

.. _known-issues-scene-partition-count-cap:

Scene partitioning is capped at 15625 partitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Affects:** ``renderer=isaacsim_rtx`` with scene partitioning enabled, and ``renderer=ovrtx``.

The underlying ``rtx.scenedb.plugin`` allocates a fixed-size pool of scene partitions and
caps it at 15625, regardless of which renderer requests them. Isaac Lab assigns one scene
partition per environment when
:attr:`~isaaclab_physx.renderers.IsaacRtxRendererCfg.enable_scene_partitioning` is enabled
for the Isaac RTX renderer, and OVRTX always assigns one scene partition per environment, so
runs with more than 15625 environments exceed the pool on either backend. Once the cap is
hit, ``rtx.scenedb.plugin`` logs a warning and discards the remaining partitions:

.. code-block:: text

    [Warning] [rtx.scenedb.plugin] SceneDbContext : Maximum number of scene partitions
    (15625) reached. Additional scene partitions will be discarded.

Environments beyond the cap are left without their own partition and end up sharing one
with another environment, so their tiled camera views can render another environment's
geometry instead of their own.

Using instanceable assets for markers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Affects:** all Kit-based renderers.

When using `instanceable assets`_ for markers, the markers do not work properly, since Omniverse does not support
instanceable assets when using the :class:`UsdGeom.PointInstancer` schema. This is a known issue and will hopefully
be fixed in a future release.

If you use an instanceable assets for markers, the marker class removes all the physics properties of the asset.
This is then replicated across other references of the same asset since physics properties of instanceable assets
are stored in the instanceable asset's USD file and not in its stage reference's USD file.

.. _instanceable assets: https://docs.isaacsim.omniverse.nvidia.com/latest/isaac_lab_tutorials/tutorial_instanceable_assets.html
.. _Omniverse Isaac Sim documentation: https://docs.isaacsim.omniverse.nvidia.com/latest/overview/known_issues.html#
.. _OpenUSD PR #4002: https://github.com/PixarAnimationStudios/OpenUSD/pull/4002


Asset import
------------

URDF importer: unresolved references for fixed joints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Affects:** the URDF importer, independent of the physics backend.

Links connected through ``fixed_joint`` elements are not merged when their URDF link entries
specify mass and inertia, even if ``merge-joint`` is set to True. This is expected behaviour —
those links are treated as full bodies rather than zero-mass reference frames.
However, the USD importer currently raises ``ReportError`` warnings showing unresolved references for such links
when they lack visuals or colliders. This is a known bug in the importer; it creates references to visuals
that do not exist. The warnings can be safely ignored until the importer is updated.


Environment and setup
---------------------

GLIBCXX errors in conda environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Affects:** conda-based installations, independent of the physics backend.

Some workflows exit with an ``OSError`` indicating ``version 'GLIBCXX_3.4.30' not found``
when running from a conda environment. The issue appears to stem from importing torch or
torch-related packages, such as tensorboard, before Isaac Sim starts. As a workaround, ensure
that all torch imports happen inside :func:`~isaaclab.app.launch_simulation`, after the runtime has
started, which should resolve the error.
