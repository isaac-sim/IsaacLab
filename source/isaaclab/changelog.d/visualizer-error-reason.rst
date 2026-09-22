Fixed
^^^^^

* Fixed ``RuntimeError: Explicitly requested visualizer(s) [...] could not be configured`` in
  :meth:`SimulationContext._resolve_visualizer_cfgs` reporting the same generic message for every
  failure. The error now includes the specific reason recorded per type (unknown visualizer type,
  ``isaaclab_visualizers`` not installed, a backend's own missing third-party dependency, or
  another import/construction failure), so the raised exception alone is enough to tell those
  cases apart instead of requiring a search through logs. The missing-package classification uses
  ``ModuleNotFoundError.name`` rather than matching text in the exception message, so an installed
  but broken visualizer package (e.g. a partially-initialized or circularly-imported module) is no
  longer misreported as simply not installed.
* Fixed ``--visualizer newton`` (the deprecated alias for ``newton_gl``) always raising this same
  ``RuntimeError`` even though it resolved successfully. :meth:`SimulationContext._resolve_visualizer_cfgs`
  compared the raw, possibly-aliased CLI string against the resolved config's canonical
  ``visualizer_type``, which never matched for an aliased request.
