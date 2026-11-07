isaaclab_contrib.custom_coupling
================================

.. automodule:: isaaclab_contrib.custom_coupling

The MJWarp and VBD classes are deprecated adapters to
:class:`~isaaclab_newton.physics.coupling.CouplerProxyCfg`. Configure named
entries and proxy mappings directly for new scenes; see
:ref:`newton-vbd-proxy-coupling`.

The adapters now use proxy feedback instead of the former shared-model penalty
reactions. Retune contact and solver parameters when migrating. Import
:mod:`isaaclab_contrib.custom_coupling.tasks` explicitly to register
``IsaacContrib-Lift-Soft-Franka-Custom-Coupling``. The environment requires a
full Isaac Lab installation containing :mod:`isaaclab_tasks`.

  .. rubric:: Classes

  .. autosummary::

    coupled_mjwarp_vbd_solver_cfg.CoupledMJWarpVBDSolverCfg
    coupled_mjwarp_vbd_solver.CoupledMJWarpVBDSolver
    franka_soft_env_cfg.FrankaSoftCustomCouplingEnvCfg

Custom Coupling
---------------

.. autoclass:: isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_solver_cfg.CoupledMJWarpVBDSolverCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_solver.CoupledMJWarpVBDSolver
  :members:
  :show-inheritance:

.. autoclass:: isaaclab_contrib.custom_coupling.franka_soft_env_cfg.FrankaSoftCustomCouplingEnvCfg
  :members:
  :show-inheritance:
