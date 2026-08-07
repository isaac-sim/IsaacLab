Added
^^^^^

* Added :class:`~isaaclab.app.LoadingScreen`, a greeting, run summary panel, and percentage progress
  bar that replaces the startup log wall on an interactive console. Each stage owns an equal slice of
  the bar and the steps reported within it advance through that slice. Startup output is spooled
  while the screen is open and replayed only when startup fails; pass ``--info`` or ``--verbose`` to
  see it on a successful run.
* Added :func:`~isaaclab.app.report_activity`, which names the startup step currently running so the
  loading screen can show it while it happens. Reports nest, so a long step keeps its label while
  its sub-steps come and go.

Changed
^^^^^^^

* Changed :class:`~isaaclab.utils.timer.Timer` to accept an ``activity`` description, reported to the
  loading screen for the duration of the timed block.
