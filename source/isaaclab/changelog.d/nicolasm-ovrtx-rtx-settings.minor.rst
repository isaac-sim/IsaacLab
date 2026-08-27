Added
^^^^^

* Added :meth:`~isaaclab.app.settings_manager.SettingsManager.get_with_prefix`, which returns the
  standalone-mode settings under a path prefix. Renderer backends that bring up their own settings system,
  such as the kit-less OVRTX renderer, use it to forward Isaac Lab's settings into that system.

Fixed
^^^^^

* Fixed :class:`~isaaclab.sensors.camera.Camera` warning that ``/rtx/rtpt/gaussian/skipTonemapping/enabled``
  had to be disabled by hand on the OVRTX backend. The setting is now applied to the RTX runtime, so
  Gaussian splats rendered through an ISP or the ``"rgb_hdr"`` output no longer need the
  ``OVRTX_rtx_rtpt_gaussian_skipTonemapping_enabled=0`` environment variable to be exported before launch.
