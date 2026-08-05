Added
^^^^^

* Enabled CUDA graph capture for coupled MPM entries that use
  capacity-bounded sparse grids.

Fixed
^^^^^

* Fixed single-world coupled MPM reset handling.
* Avoided per-step host synchronization in single-world coupled scenes without
  MPM entries.
