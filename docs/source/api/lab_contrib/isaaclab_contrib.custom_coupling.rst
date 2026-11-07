isaaclab_contrib.custom_coupling
================================

.. automodule:: isaaclab_contrib.custom_coupling

The manual MJWarp and VBD solver is deprecated and remains opt-in. Explicitly
selected manual configurations retain their shared-model one-way and two-way
penalty coupling.
For new scenes, configure named entries and proxy mappings with
:class:`~isaaclab_newton.physics.coupling.CouplerProxyCfg`; see
:ref:`newton-vbd-proxy-coupling`. Switching to proxy coupling changes the
numerical behavior and requires retuning.

Import :mod:`isaaclab_contrib.custom_coupling.tasks` explicitly to register
``IsaacContrib-Lift-Soft-Franka-Custom-Coupling``. This deprecated task alias now
uses the core generic proxy preset, including through its legacy
``physics=newton_mjwarp_vbd`` selector. It warns about the changed physics;
retune former manual-coupling policies. For new runs, use
``Isaac-Lift-Soft-Franka`` with ``physics=newton_mjwarp_vbd_proxy``. The
environment requires a full Isaac Lab installation containing
:mod:`isaaclab_tasks`.

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
