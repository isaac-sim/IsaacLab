Added
^^^^^

* Added ``scripts/environments/teleoperation/teleop_replay_agent.py``, a
  non-interactive entry point used by CI to replay captured teleop sessions
  against an Isaac Lab environment, plus a small internal
  ``isaaclab_teleop.automation`` subpackage backing it. Replaces the runtime
  patch the ``teleop-cicd`` pipeline previously applied to
  ``teleop_se3_agent.py``.

Fixed
^^^^^

* Fixed ``teleop_replay_agent.py`` driving the robot toward the world origin
  for the duration of ``--replay_start_delay_s``. The legacy
  :class:`~isaaclab.devices.openxr.OpenXRDevice` returns a default zero pose
  while the OpenXR runtime is silent, so calling ``env.step()`` during the
  start-delay window fed the Pink IK garbage targets and corrupted the robot
  pose long before real hand-tracking data flowed. The agent now registers
  ``"START"`` / ``"STOP"`` callbacks on the device -- the same path
  ``record_demos.py`` uses -- and only steps the env once the XCR replay
  dispatches the recorded ``"start"`` message through Kit's OpenXR message
  bus.
* Fixed ``teleop_replay_agent.py`` hanging the CI process when the XCR
  replay driver coroutine raised before reaching ``post_quit``. The
  previously discarded :class:`asyncio.Future` is now retained and a done
  callback logs the failure with traceback and asks Kit to quit so the
  host process exits cleanly.
* Fixed ``teleop_replay_agent.py`` leaking the USD stage when device
  construction or environment setup raised. ``env.close()`` now runs from a
  ``try/finally`` block so cleanup happens on every exit path.

Changed
^^^^^^^

* Added :paramref:`~isaaclab_teleop.automation.XcrReplayConfig.max_replay_duration_s`
  (default: ``3600``) so the completion-poll loop in
  :func:`~isaaclab_teleop.automation.start_xcr_replay` is bounded. If
  Kit's :mod:`xcr_player` ever fails to clear its private playback
  subscription, the coroutine now returns instead of spinning forever.
* Stored the :class:`omni.kit.xr.core.recorder._xr_xcr.XCRReplayAPI`
  instance in a local variable inside
  :func:`~isaaclab_teleop.automation.start_xcr_replay` so it stays alive
  for the lifetime of the replay coroutine.
