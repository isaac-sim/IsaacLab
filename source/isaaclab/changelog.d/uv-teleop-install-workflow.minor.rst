Added
^^^^^

* Added the ``isaaclab teleop`` CLI command, which groups the teleoperation workflows as
  subcommands: ``isaaclab teleop run`` for a live session
  (``scripts/environments/teleoperation/teleop_se3_agent.py``), ``isaaclab teleop record``
  for demonstration capture (``scripts/tools/record_demos.py``), and
  ``isaaclab teleop replay`` for playback (``scripts/tools/replay_demos.py``). This mirrors
  the existing ``isaaclab benchmark`` subcommand grouping, so teleoperation and
  imitation-learning workflows follow the same paradigm as reinforcement learning.
* Added the ``xr`` install extra, a one-flag aggregate of the ``isaacsim`` and ``teleop``
  extras for the XR teleoperation workflow: ``uv run --extra xr isaaclab teleop run``. Use
  ``--extra teleop`` on its own for the Isaac Teleop stack without the Kit XR runtime. ``xr``
  conflicts with ``ov``, ``viser``, ``mimic``, ``all``, and ``test``, inheriting the
  conflicts of both halves.

Fixed
^^^^^

* Fixed the ``uv`` resolution conflict that made ``uv run --extra isaacsim --extra teleop``
  unusable. ``isaacsim-kernel`` pins ``websockets==12.0`` while ``isaacteleop[cloudxr]``
  requires ``websockets>=14.0``; a ``websockets>=14.0`` override now lets the two extras
  co-resolve, so XR teleoperation can be installed from the documented ``uv`` workflow. The
  ``teleop``/``mimic`` and ``teleop``/``all`` conflicts remain because of the ``lxml`` split.
* Fixed ``ModuleNotFoundError: No module named 'isaaclab_mimic'`` when recording
  demonstrations from a teleop-only environment. ``scripts/tools/record_demos.py`` imports
  ``isaaclab_mimic`` at module level, so the ``teleop`` extra now installs the
  ``isaaclab-mimic`` package. ``robomimic`` stays in the ``mimic`` extra, since its
  ``lxml<5.0.0`` pin is what clashes with ``dex-retargeting``.
