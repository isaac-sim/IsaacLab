Fixed
^^^^^

* Fixed benign ``[omni.rtx] FindAppliedAPIPrimDefinition(...) returned nothing`` /
  ``Could not find UsdPrimDefinition for 'OmniRtx...API'`` error logs printed the first time
  the ``newton_rtx`` visualizer opens its render product, seen with ``physics=ovphysx``.
  ``launch_simulation`` now prepares the ``ovrtx`` runtime (see
  :func:`~isaaclab_ov.renderers.prepare_ovrtx_runtime`) before any physics backend gets a chance
  to touch ``pxr`` and trigger USD's (once-per-process) plug registry initialization, when
  ``newton_rtx`` is requested through ``--viz`` or a config-declared visualizer.
