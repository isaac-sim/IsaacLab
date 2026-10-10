* Fixed the Warp observation manager treating the ``history_order`` group setting as an observation term, which
  broke the Warp frontend for every task. Time-ordered histories raise :class:`NotImplementedError` in the Warp
  manager.
