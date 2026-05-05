Changed
^^^^^^^

* Changed ``--teleop_device`` default to ``None`` in ``teleop_se3_agent.py``
  and ``record_demos.py``. When omitted, the IsaacTeleop pipeline is used if
  the env configures ``isaac_teleop``; otherwise keyboard is used as fallback.
  When explicitly provided, the scripts use the legacy ``teleop_devices`` path
  and error out if no matching entry exists.
* Removed automatic ``--xr`` detection from ``--teleop_device`` containing
  ``"handtracking"``. Users who need XR with the legacy path should pass
  ``--xr`` explicitly.
