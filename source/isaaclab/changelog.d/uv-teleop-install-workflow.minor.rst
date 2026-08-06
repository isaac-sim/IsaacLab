Added
^^^^^

* Added the ``isaaclab teleop`` CLI command, which groups the teleoperation workflows as
  subcommands: ``isaaclab teleop run`` for a live session
  (``scripts/environments/teleoperation/teleop_se3_agent.py``), ``isaaclab teleop record``
  for demonstration capture (``scripts/tools/record_demos.py``), and
  ``isaaclab teleop replay`` for playback (``scripts/tools/replay_demos.py``). This mirrors
  the existing ``isaaclab benchmark`` subcommand grouping, so teleoperation and
  imitation-learning workflows follow the same paradigm as reinforcement learning.

Changed
^^^^^^^

* Changed the ``teleop`` install extra to bundle Isaac Sim, so ``uv run --extra teleop``
  installs everything the XR teleoperation workflow needs in one flag. Previously the extra
  carried only the Isaac Teleop stack and could not be combined with ``isaacsim`` at all.
  Because it now pulls Isaac Sim in, ``teleop`` conflicts with ``ov`` and ``ovphysx``;
  install those runtimes separately.

Fixed
^^^^^

* Fixed the ``uv`` resolution conflict that made Isaac Sim and Isaac Teleop impossible to
  install together. Isaac Sim 6.0 pins ``websockets``, ``coverage``, and
  ``typing-extensions`` exactly, so ``[tool.uv].override-dependencies`` relaxes all three
  and ``isaacteleop[cloudxr]``, the ``test`` extra, and the Newton viewer stack co-resolve
  with Isaac Sim. Isaac Sim 6.1 declares them as ranges, so the overrides can be dropped
  once it is published.
* Fixed ``--extra test`` being unusable with Isaac Sim, so
  ``uv run --extra teleop --extra test`` can run the teleop test suite.
* Fixed the imitation-learning training scripts being unrunnable under ``uv``. The
  ``isaacsim``/``mimic``, ``teleop``/``mimic``, ``teleop``/``all``, and ``isaacsim``/``all``
  conflicts were all stale -- ``robomimic`` no longer constrains ``lxml``, so those extras
  co-resolve and only the genuine ``packaging`` split with the OV runtimes remains. ``scripts/imitation_learning/robomimic/train.py``, ``play.py``,
  and ``robust_eval.py`` need both ``robomimic`` and the Kit runtime, and now run via
  ``uv run --extra isaacsim --extra mimic``.
* Fixed ``ModuleNotFoundError: No module named 'isaaclab_mimic'`` when recording
  demonstrations from a teleop-only environment. ``scripts/tools/record_demos.py`` imports
  ``isaaclab_mimic`` at module level, so the ``teleop`` extra now installs the
  ``isaaclab-mimic`` package. ``robomimic`` stays in the ``mimic`` extra, since its
  ``lxml<5.0.0`` pin is what clashes with ``dex-retargeting``.
