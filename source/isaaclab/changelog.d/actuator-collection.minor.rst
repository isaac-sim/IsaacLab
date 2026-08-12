Added
^^^^^

* Added :class:`~isaaclab.actuators.ActuatorCollection` as the runtime
  actuator API, with separate command, processed joint-command, and telemetry
  views.
* Added execution aggregation for disjoint stateless actuator groups while
  preserving named group configuration and access.
* Added ``joint_effort_limit`` and ``joint_velocity_limit`` as canonical
  construction-time joint-property overrides for actuator configurations.

Deprecated
^^^^^^^^^^

* Deprecated articulation-level actuator command setters and command and
  torque-telemetry properties on articulation data. Use the command view and
  ``computed_torque`` or ``applied_torque`` views on
  :attr:`~isaaclab.assets.Articulation.actuators` instead.
* Deprecated Isaac Lab execution of explicit actuator models. Enable
  :attr:`~isaaclab.sim.SimulationCfg.use_newton_actuators` to execute these
  models with Newton instead.
* Deprecated ``effort_limit_sim`` and ``velocity_limit_sim`` in favor of
  ``joint_effort_limit`` and ``joint_velocity_limit``. The deprecated names
  remain available through Isaac Lab 3.x and will be removed in 4.0.
* Deprecated group-level ``effort_limit_sim``, ``velocity_limit_sim``,
  ``armature``, ``friction``, ``dynamic_friction``, and ``viscous_friction``
  accessors. Use the corresponding :class:`~isaaclab.assets.ArticulationData`
  joint property and ``write_joint_*_to_sim_index`` writer instead; the
  accessors remain forwarding bridges through 3.x and will be removed in 4.0.
* Deprecated ``write_actuator_stiffness_to_sim`` and
  ``write_actuator_damping_to_sim``. These backend-specific writers remain
  available through 3.x; use
  :func:`~isaaclab.envs.mdp.events.randomize_actuator_gains` for managed
  randomization.

Changed
^^^^^^^

* Changed :class:`~isaaclab.actuators.ActuatorCollection` so named groups retain
  their configuration and access identity while compatible groups can share
  execution.
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

Fixed
^^^^^

* Fixed runtime, play, and startup benchmarks to step environments under
  PyTorch inference mode.
