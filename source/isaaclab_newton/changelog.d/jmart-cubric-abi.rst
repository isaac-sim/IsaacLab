Added
^^^^^

* Added runtime verification of the ``omni::cubric::IAdapter`` interface
  version in :mod:`~isaaclab_newton.physics._cubric` as defense-in-depth
  against future ABI shifts. The shim falls back to the CPU path on
  major-version mismatch or older-minor.
