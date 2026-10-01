Added
^^^^^

* Added :class:`~isaaclab_newton.physics.MjWarpActuatorBridge`, the single place Isaac Lab
  touches MuJoCo Warp's device model on behalf of a Newton actuator component. It publishes a
  component's per-step dry-friction budget into ``dof_frictionloss``, reads back the true external load on the driven DOFs
  (``-qfrc_bias + qfrc_constraint`` with the component's own friction rows removed), and can
  stiffen the friction constraint's solver reference. The Newton articulation binds it to every
  :class:`~isaaclab.actuators.BamActuatorCfg` group through Newton 1.6's ``actuator.drive`` API and rejected
  other solvers before graph capture. Writes
  go straight into the MuJoCo Warp model and are therefore not visible through Isaac Lab's
  joint-friction property; the module documents the resulting ordering contract.
  BAM viscous damping was initialized once through the articulation's joint-property setter,
  preserving it across solver property resynchronization without per-step publication.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.register_pre_actuator_callback`, an
  in-graph hook that runs immediately before the actuator step so a component can consume
  solver quantities on the same decimation iteration, and
  :meth:`~isaaclab_newton.physics.NewtonManager.register_solver_init_callback`, a one-shot hook
  that runs once the solver exists and before any CUDA graph capture. Assets initialize while
  the model is still being built, so anything that needs the concrete solver has to defer to
  the latter.
* Added a recorded deterministic native BAM pendulum trajectory with source-commit and dependency
  provenance. The regression fixture preserved the existing Newton / MJWarp behavior without a
  second servo implementation or a live upstream dependency; it did not measure upstream fidelity.
  The pendulum was provided as a standalone, authored USD fixture instead of embedded Python data.
