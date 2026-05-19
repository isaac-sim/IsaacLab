Added
^^^^^

* Added MCAP record/replay support to :class:`~isaaclab_teleop.IsaacTeleopDevice` via new
  ``mcap_record_path`` and ``mcap_replay_path`` parameters on
  :func:`~isaaclab_teleop.create_isaac_teleop_device` (mutually exclusive). ``mcap_replay_path``
  switches the underlying :class:`isacteleop.teleop_session_manager.TeleopSession` into
  :class:`SessionMode.REPLAY` and feeds the recorded tracker stream through the configured
  retargeting pipeline; ``mcap_record_path`` is a debug-grade knob that writes the live session
  to a single continuous MCAP file for pairing with the replay agent in CI. It is **not** a
  data-generation format -- the produced MCAP has no per-episode segmentation, no world-frame
  anchor state, no env reset state, and no public Python decoder.
* Added a ``--mcap_record_path`` (debug-only) flag to ``scripts/tools/record_demos.py`` that
  forwards into :func:`~isaaclab_teleop.create_isaac_teleop_device` when the IsaacTeleop stack
  is in use.
* Added ``scripts/environments/teleoperation/teleop_replay_agent.py``, a non-interactive entry
  point used by CI to replay captured Isaac Teleop sessions against an Isaac Lab environment.
  The agent gates env stepping on :func:`~isaaclab_teleop.poll_control_events` so the recorded
  START / STOP / RESET boundaries reproduce the original recording's pacing, and asks Kit to
  ``post_quit`` on the first STOP-edge after teleop has been active so the host process exits
  deterministically.

Changed
^^^^^^^

* **Breaking:** Removed the ``isaaclab_teleop.automation`` subpackage, including
  ``XcrReplayConfig`` and ``start_xcr_replay``. The XCR backend was a transitional Kit-level
  OpenXR capture/replay path that pre-dated Isaac Teleop's native MCAP record/replay. Replays
  now go through ``teleop_replay_agent.py`` against an MCAP capture produced by Isaac Teleop.
* **Breaking:** Removed the lazy legacy ``teleop_devices`` (``handtracking`` / ``manusvive``)
  accessor on
  :class:`~isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_env_cfg.PickPlaceGR1T2EnvCfg`.
  All in-tree scripts (``teleop_se3_agent.py``, ``record_demos.py``, ``teleop_replay_agent.py``)
  prefer ``env_cfg.isaac_teleop``; consumers that built the legacy
  :class:`~isaaclab.devices.openxr.OpenXRDevice` directly from the env config should construct
  it themselves or migrate to :class:`~isaaclab_teleop.IsaacTeleopDevice`.
