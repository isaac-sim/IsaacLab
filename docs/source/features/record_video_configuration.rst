.. _features_record_video:

Perspective Video Recording for Training and Benchmarking (``VideoRecorder``)
============================================================================

.. currentmodule:: isaaclab

When you enable the ``--video`` flag on RL training (e.g., ``scripts/reinforcement_learning/rsl_rl/train.py``) and benchmarking (e.g., ``scripts/benchmarks/benchmark_rsl_rl.py``) scripts, Isaac Lab captures a perspective view of the scene. This is a view separate from the tiled sensor cameras (see :ref:`overview_sensors_camera`). It uses either an Isaac Sim Kit perspective camera or a Newton GL headless viewer, depending on the active physics and renderer backends.

This page describes the configuration types, how ``camera_position`` and ``camera_target`` are filled from the environment configuration, and how frames reach each backend. For command-line usage (``--video``, intervals, log paths), see :doc:`../how-to/record_video`.


Overview
--------

* ``VideoRecorderCfg`` (``isaaclab.envs.utils.video_recorder_cfg``) holds resolution and world-space perspective parameters ``camera_position`` and ``camera_target`` (defaults to a diagonal view of the scene).
* ``VideoRecorder`` (``isaaclab.envs.utils.video_recorder``) picks a video backend from the scene (Kit vs Newton GL), builds the matching low-level capture object, and returns RGB frames via ``render_rgb_array()``.
* Direct RL, Direct MARL and manager-based RL environments copy the task's :class:`~isaaclab.envs.common.ViewerCfg` ``eye`` and ``lookat`` into those fields before the recorder is constructed, so training clips align with the task's intended viewport when ``origin_type`` is ``"world"``.


Configuration: ``VideoRecorderCfg``
-----------------------------------

The dataclass lives in ``isaaclab.envs.utils.video_recorder_cfg``. Fields ``camera_position`` and ``camera_target`` are the perspective ``eye`` and ``lookat`` points in meters.

.. literalinclude:: ../../../source/isaaclab/isaaclab/envs/utils/video_recorder_cfg.py
   :language: python
   :lines: 20-48


Task framing: ``ViewerCfg``
---------------------------

Tasks define the interactive viewer with :class:`~isaaclab.envs.common.ViewerCfg`. The ``eye`` and ``lookat`` tuples are the same values the RL base classes copy into ``VideoRecorderCfg`` (see below). If your task uses ``origin_type="world"``, those tuples are world-space positions and match what the perspective recorder expects.

.. literalinclude:: ../../../source/isaaclab/isaaclab/envs/common.py
   :language: python
   :lines: 20-28


Backend selection: Kit vs Newton GL
------------------------------------

``VideoRecorder`` resolves the implementation from the live :class:`~isaaclab.scene.InteractiveScene`: PhysX physics or Isaac RTX in the sensor stack selects the Kit path (``omni.replicator`` on ``/OmniverseKit_Persp``). The Newton GL path applies to kit-less stacks where Newton physics is paired with either the Newton Warp renderer or the OVRTX renderer (from ``isaaclab_ov``): a headless ``ViewerGL`` draws the Newton model supplied by the scene data provider. The resolver also treats ``newton_warp`` appearing among sensor renderer types as selecting this path (see the snippet below). If both Kit and Newton GL signals are present, the Kit path is chosen.

.. literalinclude:: ../../../source/isaaclab/isaaclab/envs/utils/video_recorder.py
   :language: python
   :lines: 38-59


Construction and dispatch
--------------------------

When ``env_render_mode`` is ``"rgb_array"`` (as when wrappers or scripts request RGB frames for video), the recorder instantiates the backend-specific helper and passes through ``camera_position``, ``camera_target``, and window size.

.. literalinclude:: ../../../source/isaaclab/isaaclab/envs/utils/video_recorder.py
   :language: python
   :lines: 70-114


Step 1 - Manager-based RL (``ManagerBasedRLEnv``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recorder is created inside :class:`~isaaclab.envs.manager_based_env.ManagerBasedEnv`, so ``ManagerBasedRLEnv`` must set ``cfg.video_recorder`` before ``super().__init__``.

.. literalinclude:: ../../../source/isaaclab/isaaclab/envs/manager_based_rl_env.py
   :language: python
   :start-at:        # Forward render_mode and viewer camera to VideoRecorderCfg before super().__init__()
   :end-at:            cfg.video_recorder.camera_target = tuple(float(x) for x in cfg.viewer.lookat)


Step 2 - Direct RL (``DirectRLEnv``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Direct environments build the scene first, then assign viewer fields to the recorder config and construct ``VideoRecorder`` before ``sim.reset()``.

.. literalinclude:: ../../../source/isaaclab/isaaclab/envs/direct_rl_env.py
   :language: python
   :start-at:        if self.cfg.video_recorder is not None:
   :end-at:            self.video_recorder = None


Step 3 - Direct MARL (``DirectMARLEnv``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same pattern as direct RL.

.. literalinclude:: ../../../source/isaaclab/isaaclab/envs/direct_marl_env.py
   :language: python
   :start-at:        if self.cfg.video_recorder is not None:
   :end-at:            self.video_recorder = None


Step 4 - Kit perspective (PhysX Simulation or Isaac RTX Renderer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Isaac Lab PhysX helper points the Kit perspective prim at ``camera_position`` / ``camera_target`` and attaches a Replicator RGB annotator.

.. literalinclude:: ../../../source/isaaclab_physx/isaaclab_physx/video_recording/isaacsim_kit_perspective_video.py
   :language: python
   :lines: 26-44


Step 5 - Newton GL (Newton physics with Newton Warp or OVRTX Renderer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Newton helper obtains the Newton model from the simulation's scene data provider, creates a headless ``ViewerGL``, and sets the camera from ``camera_position`` / ``camera_target`` (yaw/pitch derived from the view vector). This is the capture path for Newton physics with Newton Warp or with OVRTX rendering.

.. literalinclude:: ../../../source/isaaclab_newton/isaaclab_newton/video_recording/newton_gl_perspective_video.py
   :language: python
   :lines: 30-68


Summary
-------

.. list-table::
   :widths: 28 36 36
   :header-rows: 1

   * - Stack example
     - Video backend
     - Capture mechanism
   * - PhysX simulation + Isaac RTX renderer (Kit / full Sim)
     - Kit (``"kit"``)
     - ``/OmniverseKit_Persp`` + Replicator RGB
   * - Newton physics + Newton Warp or Newton physics + OVRTX (kit-less)
     - Newton GL (``"newton_gl"``)
     - ``newton.viewer.ViewerGL`` on the SDP Newton model


See also
--------

* :doc:`../how-to/record_video` - enable recording with ``--video`` and related flags
* :doc:`visualization` - interactive visualizers (separate from this perspective capture)
* :ref:`overview_sensors_camera` - tiled sensor cameras and renderers
