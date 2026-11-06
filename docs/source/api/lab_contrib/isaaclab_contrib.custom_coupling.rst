isaaclab_contrib.custom_coupling
================================

.. automodule:: isaaclab_contrib.custom_coupling

The custom MJWarp and VBD coupled solver is an opt-in example. Import
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
