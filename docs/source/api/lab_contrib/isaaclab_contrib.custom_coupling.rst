isaaclab_contrib.custom_coupling
================================

.. automodule:: isaaclab_contrib.custom_coupling

The custom MJWarp and VBD manager is an opt-in example. Import
:mod:`isaaclab_contrib.custom_coupling.tasks` explicitly to register
``IsaacContrib-Lift-Soft-Franka-Custom-Coupling``. The environment requires a
full Isaac Lab installation containing :mod:`isaaclab_tasks`.

  .. rubric:: Classes

  .. autosummary::

    newton_manager_cfg.CoupledMJWarpVBDSolverCfg
    coupled_mjwarp_vbd_manager.NewtonCoupledMJWarpVBDManager
    franka_soft_env_cfg.FrankaSoftCustomCouplingEnvCfg

Custom Coupling
---------------

.. autoclass:: isaaclab_contrib.custom_coupling.newton_manager_cfg.CoupledMJWarpVBDSolverCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_manager.NewtonCoupledMJWarpVBDManager
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: isaaclab_contrib.custom_coupling.franka_soft_env_cfg.FrankaSoftCustomCouplingEnvCfg
  :members:
  :show-inheritance:
