.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _physx-backend:

PhysX Backend (Isaac Sim)
=========================

The PhysX backend runs NVIDIA PhysX inside Isaac Sim, on the Omniverse Kit
runtime. Its integration lives in the ``isaaclab_physx`` package. This page
explains how that package starts Kit, owns the simulation lifecycle, configures
the PhysX scene, and exposes native PhysX data to Isaac Lab assets and sensors.
For solver selection and tuning, see :ref:`physics-backends-physx` and
:ref:`physx-solver-tuning`. For the kit-less PhysX path, see
:ref:`ovphysx-backend`.


Selecting the backend
---------------------

Configure the backend with :class:`~isaaclab_physx.physics.PhysxCfg` on
:attr:`isaaclab.sim.SimulationCfg.physics`, or select it from the command line
with ``physics=isaacsim_physx`` when the task exposes that preset:

.. code-block:: python

   from isaaclab.sim import SimulationCfg
   from isaaclab_physx.physics import PhysxCfg

   sim_cfg = SimulationCfg(physics=PhysxCfg())

The ``physx`` selector is automatic. It selects a
:class:`~isaaclab.physics.PhysxAutoCfg` that holds a ``PhysxCfg`` and, when
the task supports it, an :class:`~isaaclab_ov.physics.OvPhysxCfg`. When the
simulation is launched, the auto configuration resolves to Isaac Sim PhysX if
the run needs Kit, and otherwise to OvPhysX when one is configured. Use
``isaacsim_physx`` to require the Isaac Sim path.


Runtime: Isaac Sim and Kit
--------------------------

Launch
^^^^^^

``PhysxCfg`` names its launcher through the ``launcher_type`` class attribute,
which is ``isaaclab_physx.app:KitLauncher``.
:func:`~isaaclab.app.launch_simulation` scans the resolved configuration,
collects the launchers that the physics and renderer configurations name, and
starts them. Kit starts first. Scripts do not construct the launcher
themselves.

:class:`~isaaclab_physx.app.KitLauncher` creates the Isaac Sim
``SimulationApp``. Unless ``--experience`` is passed, it selects an Isaac Lab
experience file from the repository ``apps/`` directory based on the headless,
camera-rendering, livestream, and XR settings, for example
``isaaclab.python.headless.kit`` or ``isaaclab.python.rendering.kit``. After
startup it publishes Isaac Lab settings such as ``/isaaclab/has_gui`` that the
simulation context and renderers read. The launcher's environment variables
and arguments are documented in :mod:`isaaclab_physx.app`.

Stage
^^^^^

Isaac Lab owns the USD stage. Kit extensions, including PhysX, its tensor
views, and the viewport, read the stage from Kit's USD context instead. When
Kit is running, :class:`~isaaclab.sim.SimulationContext` registers a
:class:`~isaaclab_physx.app.KitStageBackendCfg`, and
:class:`~isaaclab_physx.app.KitStageBackend` attaches the simulation stage to
``omni.usd``'s context. Closing that backend closes the Kit stage before the
simulation clears its stage cache.

Interaction with Isaac Sim's simulation manager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~isaaclab_physx.physics.PhysxManager` is the single owner of the
simulation lifecycle. Isaac Sim's own
``isaacsim.core.simulation_manager.SimulationManager`` registers default
timeline and stage callbacks. Its stop callback invalidates the shared
``omni.physics.tensors`` simulation view, which would also invalidate the views
that Isaac Lab assets hold. Newer Isaac Sim versions can disable these default
callbacks at startup, and Isaac Lab leaves them alone in that case. On older
versions, ``PhysxManager.initialize()`` disables those callbacks and redirects
the module's ``SimulationManager`` to ``PhysxManager``. ``PhysxManager``
re-applies this patch if the Isaac Sim extension is enabled later.


Simulation lifecycle
--------------------

``PhysxManager`` implements the shared :class:`~isaaclab.physics.PhysicsManager`
contract on top of the ``omni.physx``, ``omni.physics.tensors``, and
``omni.timeline`` interfaces:

.. code-block:: text

   initialize()  Kit physics settings, PhysxSceneAPI attributes, Fabric
       │
   reset()       first call: attach stage, warm start, create the tensor view,
       │         dispatch PHYSICS_READY (assets and sensors create their views)
       │
   step()        omni.physx simulate(dt) + fetch_results()
       │
   close()       detach stage, invalidate views, release the tensor view

Initialization
   Applies Kit physics settings such as the CUDA device and GPU readback
   suppression, authors the PhysX scene attributes described in
   `Scene configuration`_, binds a default physics material, and loads or
   unloads the PhysX Fabric extension.

First reset
   Attaches the stage to PhysX (on CPU, PhysX instead loads the scene
   directly from USD), runs one warm-start update, and dispatches the
   warm-up event. It then registers a
   :class:`~isaaclab_physx.physics.PhysxBackendCfg` with the simulation
   context. The resulting ``PhysxBackend`` owns the
   ``omni.physics.tensors`` simulation view, created with the Warp frontend for
   the current stage. Finally, the manager dispatches
   ``PHYSICS_READY``, and assets and
   sensors create their typed views from the simulation view.

Step
   Advances PhysX through the ``omni.physx`` simulation interface with
   ``simulate(dt)`` followed by ``fetch_results()``. Physics stepping does not
   pump the Kit application. Rendering is driven separately by the simulation
   context. ``forward()`` refreshes articulation kinematics through the tensor
   view only when pose writes have marked them dirty.

Timeline and teardown
   ``play()``, ``pause()``, and ``stop()`` drive ``omni.timeline`` and pump
   one Kit update so timeline callbacks run synchronously. Stopping the
   timeline invalidates every tensor view. The next play or reset warms up
   PhysX and creates the views again. ``close()`` detaches PhysX from the
   stage before views are invalidated and prims are deleted.

Lifecycle events
^^^^^^^^^^^^^^^^

Backend-agnostic code registers callbacks with
:class:`~isaaclab.physics.PhysicsEvent`. ``PhysxManager`` maps those events to
its PhysX-specific :class:`~isaaclab_physx.physics.IsaacEvents`, which remain
available for backward compatibility:

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - ``PhysicsEvent``
     - ``IsaacEvents``
     - Dispatched
   * - ``MODEL_INIT``
     - ``PHYSICS_WARMUP``
     - After the warm-start update
   * - ``PHYSICS_READY``
     - ``PHYSICS_READY``
     - After the tensor view exists
   * - ``STOP``
     - ``TIMELINE_STOP``
     - When the timeline stops

``IsaacEvents`` also exposes ``PRE_PHYSICS_STEP`` and ``POST_PHYSICS_STEP``
through the ``omni.physx`` step subscription, and ``PRIM_DELETION`` during
teardown. New code should prefer ``PhysicsEvent``.


Scene configuration
-------------------

``PhysxCfg`` fields are written to the physics scene prim at
:attr:`~isaaclab.sim.SimulationCfg.physics_prim_path`. The manager applies
``PhysxSceneAPI`` and sets each remaining field as a ``physxScene:``
attribute in camel case, for example ``gpu_max_rigid_contact_count`` becomes
``physxScene:gpuMaxRigidContactCount``. Some fields are translated instead of
copied:

* ``solver_type`` becomes ``physxScene:solverType`` (``"TGS"`` or ``"PGS"``).
* ``bounce_threshold_velocity`` becomes ``physxScene:bounceThreshold``.
* The timestep becomes ``physxScene:timeStepsPerSecond``.
* GPU simulation enables GPU dynamics and the GPU broadphase. CPU simulation
  uses the MBP broadphase.
* ``enable_ccd`` takes effect only on CPU. CCD is disabled with a warning when
  GPU dynamics is enabled.
* The backend-agnostic ``deterministic`` request enables
  ``enable_enhanced_determinism``.
* Scene query support follows ``SimulationCfg.enable_scene_query_support``
  and is forced on when Kit runs with a GUI.

Per-prim physics properties are not part of ``PhysxCfg``. They are authored
through the USD schema configurations described in
:doc:`/source/concepts/schema_cfgs`. PhysX-only schema configurations, such as
:class:`~isaaclab_physx.sim.schemas.PhysxRigidBodyPropertiesCfg`, live in
:mod:`isaaclab_physx.sim.schemas`.

Fabric
^^^^^^

When :attr:`~isaaclab.sim.SimulationCfg.use_fabric` is ``True``, the default,
the manager enables the ``omni.physx.fabric`` extension. It also turns off
PhysX write-back to USD (``/physics/updateToUsd`` and related settings), so
simulated transforms reach the renderer through Fabric instead of USD. With
``use_fabric=False``, PhysX writes simulation results back to USD, which is
slower for large scenes.


Tensor views and native access
------------------------------

Every PhysX asset and sensor in ``isaaclab_physx``, except ``SurfaceGripper``,
is built on typed views created from the shared ``omni.physics.tensors``
simulation view:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Isaac Lab class
     - Native view
   * - Articulation, joint wrench sensor
     - Articulation view
   * - Rigid object, rigid object collection
     - Rigid-body view
   * - Deformable object
     - Surface or volume deformable-body view and deformable-material view
   * - Contact sensor
     - Rigid-body view and rigid-contact view
   * - IMU, frame transformer, ray caster, and similar sensors
     - Rigid-body view of the tracked bodies

Assets expose their view through ``root_view``. The contact sensor exposes its
rigid-contact view through ``contact_view``. Use
:meth:`~isaaclab_physx.physics.PhysxManager.get_physics_sim_view` to reach the
simulation view itself after the first reset. For read and write semantics,
see :doc:`/source/concepts/native-physics-api/physx`.

Environment cloning uses
:class:`~isaaclab_physx.cloner.PhysxReplicateContext`, which applies the
clone plan through the ``omni.physx`` replicator interface.


PhysX-specific and shared components
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Area
     - Shared (``isaaclab``)
     - PhysX-specific (``isaaclab_physx``)
   * - Configuration
     - :class:`~isaaclab.sim.SimulationCfg`, ``PhysicsCfg``, and
       :class:`~isaaclab.physics.PhysxAutoCfg`
     - ``PhysxCfg`` and ``PhysxBackendCfg``
   * - Runtime
     - :func:`~isaaclab.app.launch_simulation` and ``SimulationLauncher``
     - ``KitLauncher`` and ``KitStageBackend``
   * - Lifecycle
     - ``PhysicsManager`` and ``PhysicsEvent``
     - ``PhysxManager`` and ``IsaacEvents``
   * - Assets and sensors
     - Base classes, factories, and public data contracts
     - Implementations built on ``omni.physics.tensors`` views, plus
       ``SurfaceGripper``, which uses Isaac Sim's surface-gripper extension
   * - Cloning
     - Clone plan
     - ``PhysxReplicateContext``
   * - USD schemas
     - Shared schema configurations
     - ``Physx*`` schema configurations in :mod:`isaaclab_physx.sim.schemas`
   * - Scene data
     - :class:`~isaaclab.scene_data.SceneDataProvider`
     - A scene-data backend that reads poses from the tensor view or Fabric

Kit-hosted rendering, such as the
:class:`~isaaclab_physx.renderers.IsaacRtxRendererCfg` renderer, also ships in
``isaaclab_physx`` because it requires Kit. Renderer selection is
configured separately from the physics backend, but the kitless OVRTX renderer
cannot share a process with Isaac Sim PhysX. See :ref:`backend-architecture` for the
factory and registry design that all backends share.
