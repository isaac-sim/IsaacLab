Added
^^^^^

* Added :class:`~isaaclab.actuators.ActuatorCollection` as the runtime
  actuator API, with separate target-command, output-command, and telemetry
  views (:attr:`~isaaclab.actuators.ActuatorCollection.target_command` and
  :attr:`~isaaclab.actuators.ActuatorCollection.output_command`).
* Added execution aggregation for disjoint implicit actuator groups while
  preserving named group configuration and access. Explicit actuator groups
  execute one group at a time.
* Added ``actuator_effort_limit`` as the explicit actuator-model clipping
  limit, alongside the canonical ``joint_effort_limit`` and
  ``joint_velocity_limit`` joint-property overrides.
* Added ``isaaclab.actuators.newton`` hosting the Newton actuator adapter,
  host runtime, and kernels shared by every backend's native execution path.
* Added :func:`~isaaclab.actuators.newton.read_group_parameter`
  and :func:`~isaaclab.actuators.newton.write_group_parameter`
  as the single group-addressed access to Newton actuator parameters (for
  example ``("controller", "kp")`` or ``("clamping", "max_effort")``),
  implemented on Newton's selection API on every backend.

Deprecated
^^^^^^^^^^

* Deprecated articulation-level actuator command setters and command and
  torque-telemetry properties on articulation data. Use the ``target_command``
  view and ``computed_effort`` or ``applied_effort`` views on
  :attr:`~isaaclab.assets.Articulation.actuators` instead.
* Deprecated Isaac Lab execution of explicit actuator models. Enable
  :attr:`~isaaclab.sim.SimulationCfg.use_newton_actuators` to execute these
  models through the native actuator path.
* Deprecated the actuator configuration aliases ``effort_limit``,
  ``effort_limit_sim``, and ``velocity_limit_sim``, and the runtime group
  property ``effort_limit``. Use
  ``actuator_effort_limit`` for explicit actuator-model clipping and
  ``joint_effort_limit`` or ``joint_velocity_limit`` for solver limits. The
  aliases remain available through Isaac Lab 3.x and will be removed in 4.0.
* Deprecated ``write_actuator_stiffness_to_sim`` and
  ``write_actuator_damping_to_sim``. These backend-specific writers remain
  available through 3.x; use
  :func:`~isaaclab.envs.mdp.events.randomize_actuator_gains` for managed
  randomization or
  :func:`~isaaclab.actuators.newton.write_group_parameter`
  for direct controller writes.

Removed
^^^^^^^

* **Breaking:** Removed group-level ``effort_limit_sim``, ``velocity_limit_sim``,
  ``armature``, ``friction``, ``dynamic_friction``, and ``viscous_friction``
  accessors. Read the corresponding :class:`~isaaclab.assets.ArticulationData`
  joint property and use the articulation's ``write_joint_*_to_sim_index``
  writer instead.
* **Breaking:** Removed ``ArticulationData.gear_ratio`` and its backing buffers.
  The property was legacy :class:`~isaaclab.actuators.DCMotor` telemetry that
  was no longer updated by any execution path and always read one. Gear ratios
  are an actuator configuration input; read them from your actuator
  configuration instead.

Changed
^^^^^^^

* Changed :class:`~isaaclab.actuators.ActuatorCollection` so named groups retain
  their configuration and access identity while disjoint implicit groups can
  share execution.
* Changed :attr:`~isaaclab.actuators.ImplicitActuatorCfg.velocity_limit` to
  populate the actuator soft velocity-limit view. Use ``joint_velocity_limit``
  to configure the solver velocity clamp.
* Changed actuator joint-property overrides to write articulation-owned runtime
  state. Read live limits, armature, and friction through
  :class:`~isaaclab.assets.ArticulationData`; ordinary actuator groups retain
  only actuator-model state.
* **Breaking:** Changed actuator collection membership to be fixed at
  construction. Configure groups through
  :attr:`~isaaclab.assets.ArticulationCfg.actuators` before constructing the
  articulation; runtime assignment to or deletion from
  :attr:`~isaaclab.assets.Articulation.actuators` raises :class:`TypeError`.
* **Breaking:** Rejected actuator configurations that assign a joint to more
  than one group. Use disjoint joint-name expressions so each joint belongs to
  at most one actuator group.
* **Breaking:** Changed explicit actuator groups to keep the authored solver
  effort limit instead of widening it to ``1.0e9``. Effort submitted by an
  explicit model is now also clipped by the solver's ``joint_effort_limit``;
  configure it at least as large as ``actuator_effort_limit`` when the model
  should be the only clip.

Fixed
^^^^^

* Fixed runtime, play, and startup benchmarks to step environments under
  PyTorch inference mode.
