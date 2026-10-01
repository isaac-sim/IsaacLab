Added
^^^^^

* Added :class:`~isaaclab.actuators.BamActuatorCfg` and
  :class:`~isaaclab.actuators.newton.DriveBam`, a voltage-domain servo model implemented as
  Newton Warp kernels using the Newton 1.6 ``DriveBase`` API. The controller required Newton's MJWarp solver with
  :attr:`~isaaclab.sim.SimulationCfg.use_newton_actuators` enabled. It modeled firmware control,
  current limiting, motor back-EMF, supply sag, stochastic command delay and load-dependent gearbox
  friction. On MJWarp it published dry-friction and viscous-damping values to the solver and read
  external loads from the solver's generalized forces. Other Newton solvers, PhysX, OVPhysX and
  the Isaac Lab actuator loop rejected this configuration. Rotor inertia remained owned by the joint or
  :attr:`~isaaclab.actuators.ActuatorBaseCfg.armature`.
* Added :class:`~isaaclab.actuators.BamMotorCfg` for explicit motor fits, including m1/m2/m5/m6
  model selection and a local Rhoban JSON loader. BAM authoring replaced existing USD actuators
  using the configured fit and required firmware gain and nominal voltage. Recorded upstream
  motor and friction samples checked drive behavior without an upstream BAM installation.
* Added per-environment start-up sampling for supply voltage, supply sag and friction scale.
  Exposed ``vin``, ``sag_gain``, ``friction_scale``, ``kp_scale`` and ``kd_scale`` through
  :func:`~isaaclab.actuators.newton.read_group_parameter` and
  :func:`~isaaclab.actuators.newton.write_group_parameter` for event-driven randomization.
  Controller resets preserved the sampled parameters.
* Added :attr:`~isaaclab.actuators.BamActuatorCfg.stiff_frictionloss` to reduce static-friction
  creep on MJWarp with a stiff solver reference. Documented that, on this solver,
  :attr:`~isaaclab.actuators.ActuatorCollection.applied_effort` reported motor torque only and
  ``data.joint_friction`` reported the authored seed; the controller's ``friction_budget`` field
  exposed the live dry-friction budget. Inherited PD stiffness and damping were unused; the
  firmware gain was configured with :attr:`~isaaclab.actuators.BamActuatorCfg.kp_fw`.
* Used Newton 1.6's drive API throughout native actuator construction, registration, and
  parameter access. Exposed ``DriveBam`` and ``BAM_DRIVE_API`` (``NewtonBamDriveAPI`` in USD).
  Group parameter helpers accepted ``"drive"``; direct Newton access used ``actuator.drive``.
