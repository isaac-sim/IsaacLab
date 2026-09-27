Fixed
^^^^^

* Captured Newton physics and sensor CUDA graphs without executing an eager warmup. Initial steps
  and recapture preserved staged forces and callback, sensor, and solver history. Initialized
  Kamino's output state before capture instead of replaying a hidden physics step.
* Deferred physics capture until reset and decimation setup completed, and replaced the custom
  RTX capture implementation with Warp's capture API on a nonblocking stream. Unexpected physics
  capture errors were surfaced without falling back to an unaccounted eager step.
* Kept fixed MPM grids with an unbounded active-cell partition on the eager path because they
  read partition sizes on the CPU each step. Set ``max_active_cell_count`` to a positive capacity
  to enable capture for fixed grids.
