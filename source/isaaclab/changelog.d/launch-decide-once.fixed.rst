* Fixed :func:`~isaaclab.app.launch_simulation` rejecting ``--visualizer kit`` for a config that lists a
  ``newton_rtx`` visualizer: an explicit ``--visualizer`` selection drops it, so it no longer starts OVRTX.
* Fixed a Kit visualizer that an explicit ``--visualizer`` selection drops still auto-enabling cameras for its
  ``streaming_view``.
