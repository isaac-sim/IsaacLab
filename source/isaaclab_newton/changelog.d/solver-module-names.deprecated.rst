* Deprecated the ``featherstone_manager``, ``kamino_manager``, ``mjwarp_manager``, ``mpm_manager``, ``vbd_manager``
  and ``xpbd_manager`` modules of :mod:`isaaclab_newton.physics` and their ``*_manager_cfg`` modules, which now hold
  solver adapters and solver configurations. Import from the matching ``*_solver`` and ``*_solver_cfg`` modules or
  from :mod:`isaaclab_newton.physics` instead.
