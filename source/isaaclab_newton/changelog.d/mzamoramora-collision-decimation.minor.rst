Added
^^^^^

* Added :attr:`~isaaclab_newton.physics.NewtonCfg.collision_decimation` to
  re-invoke the Newton collision pipeline every ``N`` solver substeps within
  a physics tick. Defaults to ``0`` (legacy: one collide per tick). When set
  to ``0 < N < num_substeps``, the substep loop in
  :meth:`~isaaclab_newton.physics.NewtonManager._run_solver_substeps` calls
  the collision pipeline again at the matching substep boundaries so contact
  normals reflect the bodies' just-integrated poses. The last substep is
  intentionally skipped — its contact set would only feed the next tick.
  :meth:`~isaaclab_newton.physics.NewtonCfg.__post_init__` warns when
  ``collision_decimation >= num_substeps`` (the gate is silently bypassed).
