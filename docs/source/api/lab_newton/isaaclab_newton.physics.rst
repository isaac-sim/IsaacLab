isaaclab\_newton.physics
========================

.. automodule:: isaaclab_newton.physics

  .. rubric:: Classes

  .. autosummary::

    NewtonManager
    NewtonSolver
    CapturedGraph
    NewtonBackend
    NewtonCloneRecord
    StepPhase
    StepCallback
    StepGraph
    NewtonCfg
    NewtonBackendCfg
    NewtonBuilderCfg
    NewtonSoftContactCfg
    NewtonCollisionPipelineCfg
    FeatherstoneSolverAdapter
    KaminoSolverAdapter
    MPMSolverAdapter
    MJWarpSolverAdapter
    VBDSolverAdapter
    NewtonShapeCfg
    NewtonSolverCfg
    XPBDSolverAdapter
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

.. autoclass:: NewtonSolver
  :members:
  :show-inheritance:

.. autoclass:: CapturedGraph
  :members:

.. autoclass:: NewtonBackend
  :members:
  :show-inheritance:

.. autoclass:: NewtonCloneRecord
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: StepPhase
  :members:
  :show-inheritance:

.. autoclass:: StepCallback
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: StepGraph
  :members:
  :show-inheritance:

Backend Functions
-----------------

.. automodule:: isaaclab_newton.physics.newton_backend
  :members: init_solver, forward, invalidate_fk, invalidate_body_state, view_row_worlds, mark_model_changed, notify_model_changes,
    register_step_callback, unregister_step_callback, activate_actuators, add_contact_sensor, add_imu_sensor,
    build_step_graph, prepare, step, record_step, capture_graph, create_newton_backend

.. currentmodule:: isaaclab_newton.physics

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

Solver Managers
---------------

.. autoclass:: MJWarpSolverAdapter
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: VBDSolverAdapter
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: XPBDSolverAdapter
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: FeatherstoneSolverAdapter
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: KaminoSolverAdapter
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: MPMSolverAdapter
  :members:
  :inherited-members:
  :show-inheritance:
