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

.. _known-issues-closed-loop-newton:

Closed-loop articulations are not available on Newton (e.g. Agility Digit)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Affects:** ``physics=newton_mjwarp``, ``physics=newton_kamino``.

Robots whose USD encodes a closed kinematic loop — such as the achilles rod and toe push-rods
on the Agility Digit — are not currently validated on the Newton backends. The Digit-based
contrib tasks are PhysX-only and do not expose a Newton preset at all:

* ``IsaacContrib-Velocity-Flat-Digit``
* ``IsaacContrib-Velocity-Rough-Digit``
* ``IsaacContrib-Tracking-LocoManip-Digit``

Passing ``presets=newton_mjwarp`` to these tasks is rejected, because the preset never
selected a Newton backend on them; it only stripped a center-of-mass randomization from a
PhysX run. Use the default PhysX configuration for Digit-based environments.


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

.. _known-issues-animated-curve-scene-partition:

Animated curves disappear under Isaac RTX scene partitioning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Affects:** ``renderer=isaacsim_rtx`` with scene partitioning enabled. The OVRTX backend
updates animated-curve bounding boxes correctly and is unaffected.

Cables are authored as ``UsdGeom.BasisCurves`` and animated every frame. Kit RTX
computes a bounding box for a curve prim once and never refreshes it while the curve
deforms (OMPE-105749). The per-environment scene partition is sized from the union of
the bounding boxes of the prims it contains, so that union covers the cable's *initial*
extent plus whatever static geometry shares the partition. Once the cable moves outside
that union it is culled and vanishes from the tiled camera images — for example a cable
resting on a table can disappear the moment a robot arm pushes it off the edge.

The bug is specific to the Isaac RTX (Kit) backend with
:attr:`~isaaclab_physx.renderers.IsaacRtxRendererCfg.enable_scene_partitioning` enabled.

The displacement needed to trip the cull is the distance from the curve's spawn extent to the edge
of its partition's bounding box, so it depends on what else shares the partition. In a measurement
on Kit 110.1.2, the cable in ``Isaac-Lift-Cable-Franka-Camera`` vanished at 0.6 m of displacement,
while a lone curve in a partition containing only itself and a camera survived to roughly 4 m.
Smaller motions render normally, which is why a settled or lightly perturbed cable looks fine.

There are two workarounds:

* **Pin the partition bounds.** Spawn a pair of millimetre-scale static cubes at
  diagonally opposite corners of a box that conservatively envelops the environment's
  workspace. They enlarge the partition's bounding-box union to that box, so the cable
  stays inside it wherever it moves. ``Isaac-Lift-Cable-Franka`` and
  ``Isaac-Lift-Cable-Franka-Camera`` ship this workaround as the
  ``partition_bounds_marker_min`` and ``partition_bounds_marker_max`` scene entries; copy
  the pattern into custom cable environments. With the markers in place, cable visibility matches
  an unpartitioned render exactly. The markers are static visual prims without colliders, so they
  do not participate in physics, and ``Isaac-Lift-Cable-Franka-Camera`` drops them when its
  camera renderer has scene partitioning off.

* **Disable scene partitioning.** Set
  :attr:`~isaaclab_physx.renderers.IsaacRtxRendererCfg.enable_scene_partitioning` to
  ``False`` to opt out of partitioning entirely, at the cost of the per-environment
  culling.

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
torch-related packages, such as tensorboard, prior to launching ``AppLauncher``. As a
workaround, ensure that all torch imports happen after the ``AppLauncher`` instance has been
created, which should resolve the error.
