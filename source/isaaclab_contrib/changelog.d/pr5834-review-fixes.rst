Fixed
^^^^^

* Fixed :class:`~isaaclab_contrib.coupling.CouplerCfg` to reject invalid
  ownership, references, nested configs, and manager-only solver options at
  construction instead of silently building an incomplete coupled solver.

* Fixed proxy collision fallback, MPM contact and graph-capture policy, and
  Proxy/ADMM config forwarding to preserve the pinned Newton solver contract.
