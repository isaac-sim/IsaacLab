Changelog
---------

0.3.2 (2026-03-05)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_teleop.visualizers.HandJointVisualizer` to draw red sphere markers at
  each OpenXR hand joint for teleop debugging; supports single-hand or both hands when
  ``enable_visualization`` is enabled.

* Added :attr:`~isaaclab_teleop.session_lifecycle.TeleopSessionLifecycle.last_step_result` to
  expose the full pipeline output (e.g. ``hand_left``, ``hand_right``) for visualizers and
  other consumers.

* Session lifecycle now passes through additional pipeline outputs (e.g. ``hand_left``,
  ``hand_right``) when the pipeline exposes them via ``output_types``, so environments can
  enable hand visualization without changing core teleop code.

* Added ``--enable_visualization`` CLI flag to ``teleop_se3_agent.py`` and ``record_demos.py``
  to enable optional debugging visualizations (e.g. hand joint markers) for environments that
  support them.

0.3.1 (2026-02-26)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Add cleanup for Isaac Teleop session when Stop XR button is clicked and when Kit is closed.


0.3.0 (2026-02-26)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Update Isaac Teleop API usage for querying controller button states.


0.2.0 (2026-02-24)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :meth:`~isaaclab_teleop.session_lifecycle.TeleopSessionLifecycle._on_request_required_extensions` to request required
  OpenXR extensions at runtime based on Teleop devices needed for the specified environment.

0.1.0 (2026-02-18)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Initial release of the ``isaaclab_teleop`` extension.

* Added :class:`~isaaclab_teleop.IsaacTeleopDevice` providing a unified teleoperation interface
  that manages IsaacTeleop sessions, XR anchor synchronization, and retargeting pipelines within
  Isaac Lab environments.

* Added :class:`~isaaclab_teleop.IsaacTeleopCfg` for pipeline-based configuration of
  retargeting, XR anchors, and device settings directly in environment configs.
