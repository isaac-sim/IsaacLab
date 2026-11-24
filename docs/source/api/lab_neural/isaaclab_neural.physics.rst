isaaclab\_neural.physics
========================

.. automodule:: isaaclab_neural.physics

  .. rubric:: Classes

  .. autosummary::

    NewtonNeRDManager
    NeRDSolver
    MJWarpNeRDSolver
    NeRDSolverCfg
    MJWarpNeRDSolverCfg

.. currentmodule:: isaaclab_neural.physics

Physics Manager
---------------

.. autoclass:: NewtonNeRDManager
  :members: get_physics_dt, set_decimation, handles_decimation, step
  :show-inheritance:
  :exclude-members: __new__

Solvers
-------

.. autoclass:: NeRDSolver
  :show-inheritance:

.. autoclass:: MJWarpNeRDSolver
  :show-inheritance:

Solver Configuration
--------------------

.. autoclass:: NeRDSolverCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: MJWarpNeRDSolverCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__
