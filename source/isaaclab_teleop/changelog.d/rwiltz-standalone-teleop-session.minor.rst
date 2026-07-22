Added
^^^^^

* Added a standalone (non-Kit-XR) session mode to
  :class:`~isaaclab_teleop.IsaacTeleopDevice`, selected via the new
  ``use_kit_xr_bridge`` argument on
  :func:`~isaaclab_teleop.create_isaac_teleop_device`. When ``False`` the
  device bypasses the ``isaacsim.kit.xr.teleop.bridge`` extension and lets
  ``isaacteleop`` own its own OpenXR session through the CloudXR runtime, so
  teleop input/output works headless without Kit XR rendering. The
  ``record_demos.py`` and ``teleop_se3_agent.py`` teleop scripts wire this to
  the ``--xr`` flag: omitting ``--xr`` now runs IsaacTeleop for I/O only, while
  passing ``--xr`` keeps the full Kit XR rendering path unchanged.
* Added :data:`~isaaclab_teleop.CLOUDXR_STANDALONE_ENV`, a ``cloudxr-standalone.env``
  CloudXR profile that emulates a Quest 3 device so a clientless runtime advertises
  an OpenXR system (working around ``XR_ERROR_FORM_FACTOR_UNAVAILABLE``). The teleop
  scripts default ``--cloudxr_env`` to this profile when ``--xr`` is omitted (and to
  ``cloudxrjs`` when it is passed); the new ``standalone`` shorthand selects it
  explicitly.
