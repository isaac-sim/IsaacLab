Fixed
^^^^^

* Fixed ``--visualizer newton`` (the deprecated alias for ``newton_gl``) always raising
  ``RuntimeError: Explicitly requested visualizer(s) [...] could not be configured`` even though it
  resolved successfully. :meth:`SimulationContext._resolve_visualizer_cfgs` compared the raw,
  possibly-aliased CLI string against the resolved config's canonical ``visualizer_type``, which
  never matched for an aliased request.
