Changed
^^^^^^^

* Changed the ``--deterministic`` flag of :class:`~isaaclab.app.app_launcher.AppLauncher` to publish
  ``/isaaclab/render/deterministic``. A rendering backend reads this setting on initialization and
  applies its own reproducible-rendering settings; the Isaac RTX backend is the current consumer.

Removed
^^^^^^^

* **Breaking:** Removed the public ``AppLauncher.apply_rtx_determinism_settings()``. To migrate, pass
  ``--deterministic`` to :class:`~isaaclab.app.app_launcher.AppLauncher` (which now publishes
  ``/isaaclab/render/deterministic``), or call
  :func:`isaaclab_physx.renderers.isaac_rtx_renderer_utils.apply_isaac_rtx_determinism_settings` on the
  Isaac RTX backend directly.
