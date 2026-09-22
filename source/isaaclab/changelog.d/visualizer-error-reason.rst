Fixed
^^^^^

* Fixed ``RuntimeError: Explicitly requested visualizer(s) [...] could not be configured`` in
  :meth:`SimulationContext._resolve_visualizer_cfgs` reporting the same generic message for every
  failure. The error now includes the specific reason recorded per type (unknown visualizer type,
  ``isaaclab_visualizers`` not installed, or another import/construction failure), so the raised
  exception alone is enough to tell those cases apart instead of requiring a search through logs.
