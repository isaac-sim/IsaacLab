* Added kitless OVXR teleoperation and MCAP replay with independently paced rendering,
  backend-derived anchors, and ordered shutdown.
* Added ``--start_on_replay`` to the MCAP replay agent. It starts playback at startup and after
  each recorded RESET, for captures made without recorded START messages.
* Added external OVXR inputs to :class:`~isaaclab_teleop.IsaacTeleopDevice`, together with the
  ``target_frame_prim_path``, ``xr_anchor_path``, ``external_anchor_state``,
  ``required_openxr_extensions`` and ``session_cleanup_failed`` properties.
