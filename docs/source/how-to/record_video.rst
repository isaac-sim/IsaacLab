Recording video clips during training
=====================================

Isaac Lab supports recording video clips during training using the
`gymnasium.wrappers.RecordVideo <https://gymnasium.farama.org/main/_modules/gymnasium/wrappers/record_video/>`_ class.
When the ``--video`` flag is enabled, Isaac Lab captures a perspective view of the scene. The backend
is chosen automatically from the active physics and renderer stack: an Isaac Sim Kit camera or a
Newton GL headless viewer.

This feature can be enabled by installing ``ffmpeg`` and using the following command line arguments with the training
script:

* ``--video``: enables video recording during training
* ``--video_length``: length of each recorded video (in steps)
* ``--video_interval``: interval between each video recording (in steps)

Make sure to also add the ``--enable_cameras`` argument when running headless.
Note that enabling recording is equivalent to enabling rendering during training, which will slow down both startup and runtime performance.

Example usage:

.. code-block:: shell

    python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Cartpole-v0 --headless --video --video_length 100 --video_interval 500


The recorded videos will be saved in the same directory as the training checkpoints, under
``IsaacLab/logs/<rl_workflow>/<task>/<run>/videos/train``.


Overview
--------

The video recording feature is implemented using the ``VideoRecorder`` class. This class is responsible for resolving the video backend from the scene, capturing the video frames, and saving them to a file.

* ``VideoRecorderCfg`` (``isaaclab.envs.utils.video_recorder_cfg``) holds resolution and world-space
  perspective parameters ``camera_position`` and ``camera_target`` (defaults to a diagonal view of the
  scene).
* ``VideoRecorder`` (``isaaclab.envs.utils.video_recorder``) picks a video backend from the scene
  (Kit vs Newton GL), builds the matching low-level capture object, and returns RGB frames via
  ``render_rgb_array()``.
* Direct RL, Direct MARL and manager-based RL environments copy the task's
  :class:`~isaaclab.envs.common.ViewerCfg` ``eye`` and ``lookat`` into those fields before the
  recorder is constructed, so training clips align with the task's intended viewport when
  ``origin_type`` is ``"world"``.


Configuration: ``VideoRecorderCfg``
------------------------------------

The dataclass lives in ``isaaclab.envs.utils.video_recorder_cfg``. Fields ``camera_position`` and
``camera_target`` are the perspective ``eye`` and ``lookat`` points in meters.

.. literalinclude:: ../../../source/isaaclab/isaaclab/envs/utils/video_recorder_cfg.py
   :language: python
   :lines: 20-48


Task framing: ``ViewerCfg``
----------------------------

Tasks define the interactive viewer with :class:`~isaaclab.envs.common.ViewerCfg`. The ``eye`` and
``lookat`` tuples are the same values the RL base classes copy into ``VideoRecorderCfg`` (see below).
If your task uses ``origin_type="world"``, those tuples are world-space positions and match what the
perspective recorder expects.

.. literalinclude:: ../../../source/isaaclab/isaaclab/envs/common.py
   :language: python
   :lines: 20-28


Backend selection: Kit vs Newton GL
-------------------------------------

``VideoRecorder`` resolves the implementation from the live :class:`~isaaclab.scene.InteractiveScene`.
If the user provides the PhysX physics (``presets=physx,...``) or Isaac RTX (``presets=isaac_rtx_renderer,...``) in the sensor stack, the Kit path is selected (``omni.replicator`` on
``/OmniverseKit_Persp``). The Newton GL path is selected when Newton physics is active (``presets=newton,...``) or the Newton
Warp renderer (``presets=newton_renderer,...``) appears in the sensor stack - and neither PhysX nor Isaac RTX is present to claim the
Kit path. OVRTX (``presets=ovrtx_renderer,...`` from ``isaaclab_ov``) can pair with IsaacSim or Newton physics; in that case the video backend is
selected via the physics preset. If both Kit and Newton GL signals are present (e.g., ``presets=physx,isaac_rtx_renderer,...`` or ``presets=newton,newton_renderer,...``), the Kit path is chosen.

.. literalinclude:: ../../../source/isaaclab/isaaclab/envs/utils/video_recorder.py
   :language: python
   :lines: 38-59


Construction and dispatch
--------------------------

When ``env_render_mode`` is ``"rgb_array"`` (as when wrappers or scripts request RGB frames for
video), the recorder instantiates the backend-specific helper and passes through ``camera_position``,
``camera_target``, and window size.

.. literalinclude:: ../../../source/isaaclab/isaaclab/envs/utils/video_recorder.py
   :language: python
   :lines: 70-114


Step 1 - Manager-based RL (``ManagerBasedRLEnv``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recorder is created inside :class:`~isaaclab.envs.manager_based_env.ManagerBasedEnv`, so
``ManagerBasedRLEnv`` must set ``cfg.video_recorder`` before ``super().__init__``.

.. literalinclude:: ../../../source/isaaclab/isaaclab/envs/manager_based_rl_env.py
   :language: python
   :start-at:        # Forward render_mode and viewer camera to VideoRecorderCfg before super().__init__()
   :end-at:            cfg.video_recorder.camera_target = tuple(float(x) for x in cfg.viewer.lookat)


Step 2 - Direct RL (``DirectRLEnv``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Direct environments build the scene first, then assign viewer fields to the recorder config and
construct ``VideoRecorder`` before ``sim.reset()``.

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


Step 4 - Kit perspective (PhysX simulation or Isaac RTX renderer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Isaac Lab PhysX helper points the Kit perspective prim at ``camera_position`` /
``camera_target`` and attaches a Replicator RGB annotator.

.. literalinclude:: ../../../source/isaaclab_physx/isaaclab_physx/video_recording/isaacsim_kit_perspective_video.py
   :language: python
   :lines: 26-44


Step 5 - Newton GL (Newton physics with Newton Warp or OVRTX renderer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Newton helper obtains the Newton model from the simulation's scene data provider, creates a
headless ``ViewerGL``, and sets the camera from ``camera_position`` / ``camera_target``
(yaw/pitch derived from the view vector).

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
   * - Newton physics + Newton Warp or Newton physics + OVRTX
     - Newton GL (``"newton_gl"``)
     - ``newton.viewer.ViewerGL`` on the SDP Newton model


See also
--------

* :doc:`/features/visualization` - interactive visualizers
