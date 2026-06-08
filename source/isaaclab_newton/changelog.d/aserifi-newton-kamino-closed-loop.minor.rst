Added
^^^^^

* Added :class:`~isaaclab_newton.assets.articulation.closed_loop_view.ClosedLoopView`
  to simulate closed kinematic chains (parallel linkages) in maximal coordinates with
  the Kamino solver.
* Added ``max_contacts_per_world`` to :class:`~isaaclab_newton.physics.KaminoSolverCfg`
  to bound per-world contact allocation for the Kamino solver.

Fixed
^^^^^

* Fixed contact-sensor forces for the Kamino solver by routing contact aggregation
  through a unified, solver-agnostic path shared with the MuJoCo-Warp backend.
