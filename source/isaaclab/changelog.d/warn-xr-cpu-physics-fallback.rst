Changed
^^^^^^^

* Added a warning in :class:`~isaaclab.app.AppLauncher` when ``--xr`` is passed without an
  explicit ``--device``. In that case the simulation falls back to CPU physics instead of the
  usual ``cuda:0`` default. The fallback previously produced no console output, so the only way
  to discover it was to read ``AppLauncher._resolve_device_settings`` or spot ``cpu`` in the
  environment banner. The warning names the fallback and how to override it.

* Clarified the ``--device`` CLI help text to mention the XR fallback.
