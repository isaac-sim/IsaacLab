* Added kitless OVXR teleoperation, HDF5 demonstration recording and MCAP replay with independently
  paced rendering, backend-derived anchors, and ordered shutdown. ``record_demos.py`` accepts the
  live agent's ``--ovxr`` and ``--ovxr_session_mode`` options. ``replay_demos.py`` replays HDF5
  demonstrations on a kitless backend selected with ``--physics``, also when Isaac Sim is installed, and
  the MCAP replay agent replays headless on a kitless backend without ``--ovxr``, reporting each run's
  outcome without render timing; it rejects a task that sets ``target_frame_prim_path`` in that mode,
  which needs ``--ovxr`` or Kit. A kitless MCAP replay consumes exactly one recorded frame per step, so
  it runs as fast as the simulator allows.
* Added a 20-minute bound on OVXR renderer startup in live teleoperation, recording and MCAP replay.
  It covers a first start, which compiles shaders, and attachment on a GPU shared with other rendering
  processes.
* Added ``--ovxr_step_hz`` to the live teleoperation agent and the demonstration recorder. It caps
  the host-control rate with ``--ovxr`` while XR rendering runs independently.
* Added ``--start_on_replay`` to the MCAP replay agent. It starts playback at startup and after
  each recorded RESET, for captures made without recorded START messages.
* Added external OVXR inputs to :class:`~isaaclab_teleop.IsaacTeleopDevice`, together with the
  ``target_frame_prim_path``, ``xr_render_anchor_path``, ``external_anchor_state``,
  ``required_openxr_extensions`` and ``session_cleanup_failed`` properties.
* Added the experimental ``prepare_environment`` task hook to :class:`~isaaclab_teleop.IsaacTeleopCfg`.
  The live, replay and recording launchers run it once before creating simulation resources.
* Added :meth:`~isaaclab_teleop.IsaacTeleopDevice.notify_reset`, which reports an environment reset
  to the input pipeline: retargeters reset on their next execution and actions computed before the
  reset are withheld, while operator control events are preserved.
* Added OVXR launch validation that rejects dynamic ``CUSTOM`` XR anchors, whose callback needs a
  previous head pose that OVXR does not supply. Kit keeps supporting them.
* Added terminal handling of retargeting failures in live OVXR sessions. The borrowed XR session keeps
  its attached input actions, so the failure ends the session instead of restarting input on it.
