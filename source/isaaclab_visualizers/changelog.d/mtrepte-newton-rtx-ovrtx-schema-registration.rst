Fixed
^^^^^

* Fixed benign ``[omni.rtx] FindAppliedAPIPrimDefinition(...) returned nothing`` /
  ``Could not find UsdPrimDefinition for 'OmniRtx...API'`` error logs printed when the
  ``newton_rtx`` visualizer opens its first render product. ``NewtonViewerRTX`` now also prepares
  the ``ovrtx`` runtime before Newton's ``ViewerRTX`` imports ``ovrtx``, as a defense-in-depth
  fallback for scripts that construct a viewer without going through
  ``isaaclab.app.sim_launcher.launch_simulation`` (see the ``isaaclab`` changelog for the
  primary fix).
