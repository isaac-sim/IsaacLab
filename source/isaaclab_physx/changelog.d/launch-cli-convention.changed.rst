* :class:`~isaaclab_physx.app.KitLauncher` reads whether the run has the Kit visualizer from the ``visualizer``
  selection (``--visualizer kit``) instead of the removed ``kit_visualizer`` launcher argument.
* **Breaking:** :class:`~isaaclab_physx.app.KitLauncher` no longer reads a ``headless`` launcher argument: it opens
  a window exactly when ``--visualizer`` selects ``kit`` and neither ``HEADLESS=1`` nor livestreaming is set, so a
  Kit visualizer only a video records from (``--video viz:kit``) runs headless. XR auto-starts under the same
  rule, when ``--visualizer`` does not select ``kit``.
