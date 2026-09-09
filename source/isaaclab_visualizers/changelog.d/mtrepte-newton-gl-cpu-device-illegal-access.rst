Fixed
^^^^^

* Fixed ``NewtonVisualizationMarkers.render`` and ``_ensure_mesh_registered`` allocating Warp
  marker/mesh arrays on Warp's process-global default device instead of the marker/sim device.
  Combined with ``--device cpu`` on a machine that also has a CUDA device present, this caused a
  CUDA illegal memory access when the ``newton_gl`` visualizer rendered marker overlays.
  ``_ensure_mesh_registered`` now takes an explicit ``device`` argument.
