Added
^^^^^

* Added :meth:`~isaaclab.sim.SimulationContext.request_reset` and
  :meth:`~isaaclab.sim.SimulationContext.consume_reset_request` to
  :class:`~isaaclab.sim.SimulationContext` for routing visualizer-initiated episode resets.
* Added :meth:`~isaaclab.visualizers.BaseVisualizer.is_reset_requested` and
  :meth:`~isaaclab.visualizers.BaseVisualizer.consume_reset_request` to
  :class:`~isaaclab.visualizers.BaseVisualizer` so all backends expose a consistent
  reset-request API.
* Added **Reset Episode** button to the Kit environment window
  (:class:`~isaaclab.envs.ui.BaseEnvWindow`).

Fixed
^^^^^

* Fixed :func:`~isaaclab.utils.dict.class_to_dict` silently returning a
  :class:`~isaaclab.utils.string.ResolvableString` instance (rather than a plain :class:`str`)
  when the value appeared inside a tuple or list, causing ``OmegaConf.create`` to raise
  ``UnsupportedValueType`` for fields such as ``cloning_contexts``.
