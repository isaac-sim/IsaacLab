* Deprecated the Kit visualizer; see :class:`~isaaclab_visualizers.kit.KitBackendCfg`. Livestreaming and XR still
  start Kit, so the Kit visualizer that they add or that is already configured does not emit the deprecation
  warning.
* Deprecated the ``isaacsim_physx`` and ``isaacsim_rtx`` selectors, which run on Isaac Sim (Kit). Selecting them by
  ``physics=``, ``renderer=`` or ``presets=`` still works but emits a :class:`FutureWarning` naming the Kit-free
  alternatives (``ovphysx`` or ``newton_mjwarp``, and ``ovrtx`` or ``newton_renderer``).
