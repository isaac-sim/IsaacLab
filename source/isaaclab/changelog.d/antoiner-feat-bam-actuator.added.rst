* Added :class:`~isaaclab.actuators.BamActuatorCfg` and
  :class:`~isaaclab.actuators.newton.DriveBam`, a voltage-domain servo model implemented as
  Newton Warp kernels using the Newton 1.6 ``DriveBase`` API. The controller required Newton's MJWarp solver with
  :attr:`~isaaclab.sim.SimulationCfg.use_newton_actuators` enabled. It modeled firmware control,
  current limiting, motor back-EMF, supply sag, stochastic command delay and load-dependent gearbox
  friction. On MJWarp it published the dry-friction budget each step, initialized passive joint
  damping once from the motor fit, and read
  external loads from the solver's generalized forces. Other Newton solvers, PhysX, OVPhysX and
  the Isaac Lab actuator loop rejected this configuration. Rotor inertia remained owned by the joint or
  :attr:`~isaaclab.actuators.ActuatorBaseCfg.armature`.
  Core validation used the backend-neutral dispatcher without importing the optional Newton backend package.
  Motor effort defaulted to the maximum configured supply voltage times ``kt / resistance``;
  an explicit ``actuator_effort_limit`` overrode this default without changing the joint effort limit.
  Load-dependent friction used the previous clamped drive output, while supply sag retained the
  previous unclamped motor torque.
* Added :class:`~isaaclab.actuators.BamMotorCfg` for explicit motor fits, including m1/m2/m5/m6
  model selection and coefficients configured directly in Python. BAM authoring replaced existing USD actuators
  using the configured fit and required firmware gain and nominal voltage. It authored a positive
  joint-friction seed to allocate the solver constraint even with zero Coulomb friction. Recorded upstream
  motor and friction samples checked drive behavior without an upstream BAM installation.
  Component tests loaded a standalone, authored two-servo USD fixture; configuration authoring
  remained covered by dedicated tests.
  Reference recordings also covered stateful supply sag from BAM ``62bd8ce`` and mjlab 1.3.0
  command-buffer output for 3--6-step delays, including a partial reset. Delay tests replayed
  recorded lag draws independently of Warp's RNG.
* Added per-environment start-up sampling for supply voltage, supply sag and friction scale.
  Exposed ``vin``, ``sag_gain``, ``friction_scale``, ``kp_scale`` and ``kd_scale`` through
  :func:`~isaaclab.actuators.newton.read_group_parameter` and
  :func:`~isaaclab.actuators.newton.write_group_parameter` for event-driven randomization.
  Controller resets preserved the sampled parameters. Command lag and phase were shared by the group's
  joints within each environment. Delay resets advanced the selected environments' random streams,
  including during CUDA graph replay, without changing untouched environments.
* Added :attr:`~isaaclab.actuators.BamActuatorCfg.stiff_frictionloss` to reduce static-friction
  creep on MJWarp with a stiff solver reference. Documented that, on this solver,
  :attr:`~isaaclab.actuators.ActuatorCollection.applied_effort` reported motor torque only and
  ``data.joint_friction`` reported the authored seed; the controller's ``friction_budget`` field
  exposed the live dry-friction budget. Inherited PD stiffness and damping were unused; the
  firmware gain was configured with :attr:`~isaaclab.actuators.BamActuatorCfg.kp_fw`.
* Used Newton 1.6's drive API throughout native actuator construction, registration, and
  parameter access. Exposed ``DriveBam`` and ``BAM_DRIVE_API`` (``NewtonBamDriveAPI`` in USD).
  Group parameter helpers accepted ``"drive"``; direct Newton access used ``actuator.drive``.
