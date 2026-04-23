Recording video clips during training
=====================================

Isaac Lab supports recording video clips during training using the
`gymnasium.wrappers.RecordVideo <https://gymnasium.farama.org/main/_modules/gymnasium/wrappers/record_video/>`_ class.
When the ``--video`` flag is enabled, Isaac Lab captures a video of the scene. Two modes are supported:

* **Perspective** (default) - captures a single, wide-angle view of the scene from a configurable, world-space camera position.
* **Tiled** - captures all (or a subset of) parallel environments laid out in a grid, using
  ``Camera`` sensors already in the scene or a fallback camera spawned automatically.

The backend is chosen automatically from the active physics and renderer stack: an Isaac Sim Kit
camera (PhysX / Isaac RTX) or a Newton GL headless viewer (Newton / Newton Warp).

This feature can be enabled by installing ``ffmpeg`` and using the following command line arguments with the training
script:

* ``--video``: enables perspective video recording during training (equivalent to ``--video=perspective``)
* ``--video=tiled``: enables tiled grid video recording (all parallel environments in one frame)
* ``--video_length``: length of each recorded video (in steps)
* ``--video_interval``: interval between each video recording (in steps)

Note that enabling recording is equivalent to enabling rendering during training, which will slow down both startup and runtime performance.

Example usage (perspective):

.. code-block:: shell

    python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Cartpole-v0 --headless --video --video_length 100 --video_interval 500

Example usage (tiled):

.. code-block:: shell

    # all envs per frame
    python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Cartpole-v0 --headless --video=tiled --video_length 100 --video_interval 500
    # or a subset of envs per frame
    python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Cartpole-v0 --headless --video=tiled "env.video_recorder.video_num_tiles=9" --video_length 100 --video_interval 500

The recorded videos will be saved in the same directory as the training checkpoints, under
``IsaacLab/logs/<rl_workflow>/<task>/<run>/videos/train``.


Overview
--------

The video recording feature is implemented using the ``VideoRecorder`` class. This class is responsible for resolving the video backend from the scene, capturing the video frames, and saving them to a file.

* ``VideoRecorderCfg`` (``isaaclab.envs.utils.video_recorder_cfg``) holds the recording mode
  (``video_mode``), resolution, world-space perspective parameters (``camera_position``,
  ``camera_target``), and tiled-mode parameters (``video_num_tiles``, ``fallback_camera_cfg``).
* ``VideoRecorder`` (``isaaclab.envs.utils.video_recorder``) picks a video backend from the scene
  (Kit vs Newton GL), builds the matching low-level capture object, and returns RGB frames via
  ``render_rgb_array()``.
* Direct RL, Direct MARL and manager-based RL environments copy the task's
  :class:`~isaaclab.envs.common.ViewerCfg` ``eye`` and ``lookat`` into those fields before the
  recorder is constructed, so perspective training clips align with the task's intended viewport when
  ``origin_type`` is ``"world"``.


Configuration: ``VideoRecorderCfg``
------------------------------------

The dataclass lives in ``isaaclab.envs.utils.video_recorder_cfg``. Key fields:

* ``video_mode`` - ``"perspective"`` (default) or ``"tiled"``.
* ``camera_position`` / ``camera_target`` - world-space eye and look-at points (metres) for
  perspective mode.
* ``video_num_tiles`` - maximum environments per tiled frame; ``-1`` means all.
* ``fallback_camera_cfg`` - a :class:`~isaaclab.sensors.camera.CameraCfg` spawned automatically
  when no suitable ``Camera`` exists in the scene (tiled mode only).

.. literalinclude:: ../../../source/isaaclab/isaaclab/envs/utils/video_recorder_cfg.py
   :language: python
   :lines: 25-91


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
   :lines: 33-55


Construction and dispatch
--------------------------

When ``env_render_mode`` is ``"rgb_array"`` (as when wrappers or scripts request RGB frames for
video), the recorder checks ``video_mode`` and instantiates the backend-specific helper accordingly.
Perspective mode passes through ``camera_position``, ``camera_target``, and window size.
Tiled mode resolves or spawns a ``Camera`` sensor and tiles all requested environments.

.. literalinclude:: ../../../source/isaaclab/isaaclab/envs/utils/video_recorder.py
   :language: python
   :lines: 57-161


Customising the camera view
----------------------------

When ``--video`` (perspective) is passed, the recording camera uses the same
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


Tiled recording
---------------

When ``--video=tiled`` is passed, every frame in the output video contains all parallel environments
tiled into a single image. This is useful for visually inspecting policy behaviour across many
environments simultaneously.

**How it works:**

1. ``VideoRecorder`` selects the backend (Kit or Newton GL) as usual.
2. It looks for an existing ``Camera`` sensor in the scene with a supported renderer
   (Isaac RTX or OV RTX for Kit; Newton Warp for Newton GL).
3. If none is found, it spawns the ``fallback_camera_cfg`` (default:
   :data:`~isaaclab.envs.utils.video_recorder_cfg.DEFAULT_TILED_RECORDING_CAMERA_CFG`) - a
   pinhole camera placed at ``(-7, 0, 3)`` m, angled ~12° downward, covering the first environment.
4. Tile count is controlled by ``video_num_tiles`` (``-1`` = all environments).

**Using a task's own Camera sensor:**

If your task already declares a ``Camera`` with an Isaac RTX or OV RTX renderer (for Kit
backends), that sensor is reused automatically and no fallback camera is spawned. Set
``fallback_camera_cfg=None`` to enforce this:

.. code-block:: python

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    video_recorder = VideoRecorderCfg(
        video_mode="tiled",
        fallback_camera_cfg=None,  # require an existing Camera in the scene
    )

**Custom fallback camera:**

To customise the auto-spawned tiled camera, override ``fallback_camera_cfg`` in the task config:

.. code-block:: python

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg, DEFAULT_TILED_RECORDING_CAMERA_CFG

    video_recorder = VideoRecorderCfg(
        video_mode="tiled",
        video_num_tiles=9,
        fallback_camera_cfg=DEFAULT_TILED_RECORDING_CAMERA_CFG.replace(
            offset=DEFAULT_TILED_RECORDING_CAMERA_CFG.OffsetCfg(
                pos=(-10.0, 0.0, 5.0), rot=(0.0, 0.1045, 0.0, 0.9945), convention="world"
            )
        ),
    )


Summary
-------

.. list-table::
   :widths: 20 20 22 38
   :header-rows: 1

   * - Mode (``--video=``)
     - Stack example (``presets=...``)
     - Video backend
     - Capture mechanism
   * - ``perspective`` (default)
     - ``physx,...`` or ``isaac_rtx_renderer,...``
     - Kit (``"kit"``)
     - ``/OmniverseKit_Persp`` + Replicator RGB
   * - ``perspective`` (default)
     - ``newton,...`` or ``newton_renderer,...`` (no Kit signals)
     - Newton GL (``"newton_gl"``)
     - ``newton.viewer.ViewerGL`` on the SDP Newton model
   * - ``tiled``
     - ``physx,...`` or ``isaac_rtx_renderer,...``
     - Kit (``"kit"``)
     - Isaac RTX ``Camera`` sensor grid
   * - ``tiled``
     - ``newton,...`` or ``newton_renderer,...`` (no Kit signals)
     - Newton GL (``"newton_gl"``)
     - Newton Warp ``Camera`` sensor grid


See also
--------

* :doc:`/source/features/visualization` - interactive visualizers
