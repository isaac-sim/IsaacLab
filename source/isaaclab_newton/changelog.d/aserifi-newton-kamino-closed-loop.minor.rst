Added
^^^^^

* Added ``max_contacts_per_world`` to :class:`~isaaclab_newton.physics.KaminoSolverCfg`
  to bound per-world contact allocation for the Kamino solver.

Fixed
^^^^^

* Fixed contact-sensor forces for the Kamino solver by routing contact aggregation
  through a unified, solver-agnostic path shared with the MuJoCo-Warp backend.
