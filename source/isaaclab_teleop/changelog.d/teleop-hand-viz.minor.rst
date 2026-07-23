Added
^^^^^

* Added optional tracking debug visualization for IsaacTeleop sessions. Red sphere markers
  are rendered at each OpenXR hand joint and RGB axis markers at the controller aim poses
  when the ``enable_debug_visualization`` argument of
  :func:`~isaaclab_teleop.create_isaac_teleop_device` is set (exposed as the
  ``--enable_debug_visualization`` CLI flag on the teleoperation scripts).

Fixed
^^^^^

* Fixed teleop session restart churn when the retargeting pipeline raises during a step
  (e.g. on degenerate tracking data): the failure is now diagnosed against the actual Kit
  XR session state instead of always being reported as an external XR teardown, and session
  re-creation is rate-limited to once per second. External Stop-AR/Start-AR recovery latency
  is unchanged.
