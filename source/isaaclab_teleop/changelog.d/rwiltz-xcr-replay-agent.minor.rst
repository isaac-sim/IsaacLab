Added
^^^^^

* Added ``scripts/environments/teleoperation/teleop_replay_agent.py``, a
  non-interactive entry point used by CI to replay captured teleop sessions
  against an Isaac Lab environment, plus a small internal
  ``isaaclab_teleop.automation`` subpackage backing it. Replaces the runtime
  patch the ``teleop-cicd`` pipeline previously applied to
  ``teleop_se3_agent.py``.
