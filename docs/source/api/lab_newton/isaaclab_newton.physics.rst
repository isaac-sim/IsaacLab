isaaclab\_newton.physics
========================

.. automodule:: isaaclab_newton.physics

  .. rubric:: Classes

  .. autosummary::

    NewtonManager
    NewtonBackend
    NewtonCloneRecord
    MjWarpActuatorBridge
    NewtonCfg
    NewtonBackendCfg
    NewtonBuilderCfg
    NewtonSoftContactCfg
    NewtonCollisionPipelineCfg
    NewtonShapeCfg
    NewtonSolver
    MJWarpSolver
    VBDSolver
    XPBDSolver
    FeatherstoneSolver
    KaminoSolver
    MPMSolver
    NewtonSolverCfg
    MJWarpSolverCfg
    VBDSolverCfg
    XPBDSolverCfg
    FeatherstoneSolverCfg
    KaminoCollisionDetectorCfg
    KaminoConstraintsCfg
    KaminoDVICfg
    KaminoDVISolverCfg
    KaminoDynamicsCfg
    KaminoFKCfg
    KaminoMaterialsCfg
    KaminoPADMMCfg
    KaminoPADMMSolverCfg
    MPMSolverCfg
    HydroelasticSDFCfg

.. currentmodule:: isaaclab_newton.physics

Physics Manager
---------------

.. autoclass:: NewtonManager
  :members:
  :inherited-members:
  :show-inheritance:

.. autofunction:: create_builder

.. autoclass:: NewtonBackend
  :members:
  :exclude-members: __init__

.. autoclass:: NewtonCloneRecord
  :members:

Solvers
-------

.. autoclass:: NewtonSolver
  :members:

.. autoclass:: MJWarpSolver
  :show-inheritance:

.. autoclass:: VBDSolver
  :show-inheritance:

.. autoclass:: XPBDSolver
  :show-inheritance:

.. autoclass:: FeatherstoneSolver
  :show-inheritance:

.. autoclass:: KaminoSolver
  :show-inheritance:

.. autoclass:: MPMSolver
  :show-inheritance:

Actuator Solver Access
----------------------

.. autoclass:: MjWarpActuatorBridge
  :members:
  :show-inheritance:

Physics Configuration
---------------------

.. autoclass:: NewtonCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: NewtonBackendCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: NewtonBuilderCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autofunction:: create_newton_builder

.. autoclass:: NewtonSoftContactCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: NewtonSolverCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: MJWarpSolverCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: VBDSolverCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: XPBDSolverCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: FeatherstoneSolverCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: KaminoPADMMCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: KaminoDVICfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: KaminoDynamicsCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: KaminoConstraintsCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: KaminoFKCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: KaminoCollisionDetectorCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: KaminoMaterialsCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: KaminoPADMMSolverCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: KaminoDVISolverCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: MPMSolverCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: NewtonCollisionPipelineCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: HydroelasticSDFCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: NewtonShapeCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

MPM Solver State
----------------

.. autofunction:: reset_mpm_solver_state
