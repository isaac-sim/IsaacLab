isaaclab\_newton.physics
========================

.. automodule:: isaaclab_newton.physics

  .. rubric:: Classes

  .. autosummary::

    NewtonManager
    SolverSpec
    CapturedGraph
    NewtonBackend
    NewtonCloneRecord
    StepCallbacks
    StepCallback
    StepGraph
    MjWarpActuatorBridge
    NewtonCfg
    NewtonBackendCfg
    NewtonBuilderCfg
    NewtonSoftContactCfg
    NewtonCollisionPipelineCfg
    NewtonShapeCfg
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

.. autoclass:: CapturedGraph
  :members:

.. autoclass:: NewtonBackend
  :members:
  :exclude-members: __init__

.. autoclass:: NewtonCloneRecord
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: StepCallbacks
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
  :members: init_solver, forward, invalidate_fk, invalidate_body_state, mark_model_changed, notify_model_changes,
    register_step_callback, unregister_step_callback, activate_actuators, add_contact_sensor, add_imu_sensor,
    build_step_graph, prepare, step, capture_graph, run_query, refit_bvh

.. currentmodule:: isaaclab_newton.physics

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

Solver Specifications
---------------------

.. autoclass:: SolverSpec
  :members:

.. autofunction:: solver_spec

.. autofunction:: build_from_cfg

.. autofunction:: register_builder_attributes

.. autofunction:: usd_schema_resolvers

.. autodata:: MJWARP_SPEC
  :annotation:

.. autodata:: VBD_SPEC
  :annotation:

.. autodata:: XPBD_SPEC
  :annotation:

.. autodata:: FEATHERSTONE_SPEC
  :annotation:

.. autodata:: KAMINO_SPEC
  :annotation:

.. autodata:: MPM_SPEC
  :annotation:

.. autofunction:: reset_mpm_solver_state
