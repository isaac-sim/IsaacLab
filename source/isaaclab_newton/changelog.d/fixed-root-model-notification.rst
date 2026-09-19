Fixed
^^^^^

* Fixed Newton-backed articulation and rigid-object root-pose writes leaving solver-owned
  model transforms stale. Nonfloating root writes now notify the solver immediately,
  including masked writes and writes that defer forward kinematics with ``skip_forward``.
