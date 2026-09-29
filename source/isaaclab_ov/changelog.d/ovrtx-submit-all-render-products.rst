Fixed
^^^^^

* Fixed a large rendering slowdown in the OVRTX renderer by submitting every registered
  render product on each step instead of only the requested ones. OVRTX batches products
  within a step, so naming a subset made every product that re-entered the set on a later
  step far more expensive. Only the requested cameras are read back, so products left out
  of a request keep their previous outputs. Both the legacy and ``ovstage`` submission
  paths were updated.
