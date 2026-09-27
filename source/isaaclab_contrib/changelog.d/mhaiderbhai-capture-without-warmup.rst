Fixed
^^^^^

* Applied Newton's deferred capture lifecycle to coupled solvers and kept fixed MPM entries with
  an unbounded active-cell partition eager. Set the entry's ``max_active_cell_count`` to a positive
  capacity to enable capture for fixed grids.
