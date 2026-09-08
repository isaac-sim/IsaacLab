Added
^^^^^

* Added :func:`~isaaclab_ov.renderers.prepare_ovrtx_runtime`, a best-effort helper that patches
  ``ovrtx``'s native library search paths and registers its USD schema/plugin paths with USD's
  plug registry. Used by ``isaaclab.app.sim_launcher.launch_simulation`` and
  ``NewtonViewerRTX`` to avoid benign ``FindAppliedAPIPrimDefinition(...) returned nothing``
  error logs when the ``newton_rtx`` visualizer is used.
