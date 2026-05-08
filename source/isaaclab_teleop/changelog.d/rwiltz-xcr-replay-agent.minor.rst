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
