Changed
^^^^^^^

* Changed :meth:`~isaaclab.envs.DirectRLEnv.set_debug_vis` and
  :meth:`~isaaclab.envs.DirectMARLEnv.set_debug_vis`, and the
  :class:`~isaaclab.ui.widgets.ManagerLiveVisualizer` debug visualization toggles, to register
  their callbacks through the simulation context's visualization marker registry instead of the
  deprecated Kit ``IApp.get_post_update_event_stream`` API. This matches how assets, sensors and
  the managers already register. Debug visualization callbacks now run when a marker-capable
  visualizer dispatches them, rather than on every Kit post-update tick, so they no longer run
  when nothing is drawing them.

Fixed
^^^^^

* Fixed debug visualization failing in kitless mode. Enabling it raised
  ``NameError: name 'omni' is not defined`` because ``omni.kit.app`` is imported only when Kit is
  present but was used unconditionally. The registry path has no Kit dependency.
