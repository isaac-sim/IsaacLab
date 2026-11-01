* Added kitless OVXR teleoperation and MCAP replay with independently paced rendering,
  backend-derived anchors, and ordered shutdown.
* Preserved recorded replay controls by default, with explicit ``--start_on_replay`` for recordings without start events.
* Added task hooks to :class:`~isaaclab_teleop.IsaacTeleopCfg`: ``prepare_environment``,
  ``action_conditioner``, ``action_validator`` and ``require_initial_bimanual_hand_tracking``.
* Added ``render_anchor_pos`` and ``render_anchor_rot`` to :class:`~isaaclab_teleop.XrCfg` for a
  render-only anchor offset.
* Added external OVXR inputs to :class:`~isaaclab_teleop.IsaacTeleopDevice`, together with the
  ``target_frame_prim_path``, ``xr_render_anchor_path``, ``external_anchor_state``,
  ``required_openxr_extensions``, ``session_cleanup_failed`` and ``hand_tracking_presence``
  properties.
