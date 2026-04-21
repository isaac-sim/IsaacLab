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


Customising the camera view
----------------------------

When ``--video`` is passed, the recording camera uses the same
position and look-at target as the interactive viewer. The defaults come from
:class:`~isaaclab.envs.common.ViewerCfg`:

* ``eye = (7.5, 7.5, 7.5)`` — camera position in world space (metres)
* ``lookat = (0.0, 0.0, 0.0)`` — camera look-at target in world space (metres)
* Resolution ``1280x720``

To change the recording angle, override the ``viewer`` field in your task's environment config.
The RL base classes automatically copy ``eye`` and ``lookat`` into ``VideoRecorderCfg`` before
recording starts (when ``origin_type`` is ``"world"``), so the video clip uses the same viewpoint
as the interactive viewport:

.. code-block:: python

    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.envs.common import ViewerCfg
    from isaaclab.utils import configclass

    @configclass
    class MyTaskCfg(ManagerBasedRLEnvCfg):
        viewer: ViewerCfg = ViewerCfg(
            eye=(5.0, 5.0, 5.0),
            lookat=(0.0, 0.0, 1.0),
        )


Summary
-------

.. list-table::
   :widths: 40 22 38
   :header-rows: 1

   * - Stack example (``presets=...``)
     - Video backend
     - Capture mechanism
   * - ``physx,...`` or ``isaac_rtx_renderer,...``
     - Kit (``"kit"``)
     - ``/OmniverseKit_Persp`` + Replicator RGB
   * - ``newton,...`` or ``newton_renderer,...`` (no Kit signals)
     - Newton GL (``"newton_gl"``)
     - ``newton.viewer.ViewerGL`` on the SDP Newton model
   * - ``newton,...,ovrtx_renderer,...`` (OVRTX + Newton physics)
     - Newton GL (``"newton_gl"``)
     - ``newton.viewer.ViewerGL`` on the SDP Newton model


See also
--------

* :doc:`/source/features/visualization` - interactive visualizers
