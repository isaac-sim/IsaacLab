Added
^^^^^

* Added the ``isaaclab teleop``, ``isaaclab record``, and ``isaaclab replay`` CLI commands,
  which run ``scripts/environments/teleoperation/teleop_se3_agent.py``,
  ``scripts/tools/record_demos.py``, and ``scripts/tools/replay_demos.py`` respectively. These
  mirror the existing ``isaaclab train`` and ``isaaclab play`` entry points, so teleoperation
  and imitation-learning workflows follow the same ``uv run`` paradigm as reinforcement
  learning.

Fixed
^^^^^

* Fixed the ``uv`` resolution conflict that made ``uv run --extra isaacsim --extra teleop``
  unusable. ``isaacsim-kernel`` pins ``websockets==12.0`` while ``isaacteleop[cloudxr]``
  requires ``websockets>=14.0``; a ``websockets>=14.0`` override now lets the two extras
  co-resolve, so XR teleoperation can be installed from the documented ``uv`` workflow. The
  ``teleop``/``mimic`` and ``teleop``/``all`` conflicts remain because of the ``lxml`` split.
