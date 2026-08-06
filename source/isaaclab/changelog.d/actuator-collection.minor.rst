Added
^^^^^

* Added :class:`~isaaclab.actuators.ActuatorCollection` as the runtime
  actuator API, with separate command and processed joint-command views,
  and telemetry staging.
* Added execution aggregation for disjoint stateless actuator groups while
  preserving named group configuration and access.
* Added reusable Warp execution for implicit and stateless explicit actuator
  batches to avoid per-step staging allocations and launch reconstruction.

Deprecated
^^^^^^^^^^

* Deprecated articulation-level actuator command setters and actuator command
  properties on articulation data in favor of the command view on
  :attr:`~isaaclab.assets.Articulation.actuators`.

Changed
^^^^^^^

* Changed :class:`~isaaclab.actuators.ActuatorCollection` ownership so it stages
  routing, commands, and telemetry while execution actuators own model
  parameters, scratch tensors, and outputs. Compatible configured groups retain
  their concrete public identities while sharing stable execution storage.
* **Breaking:** Changed actuator collection membership to be fixed at
  construction. Configure groups through
  :attr:`~isaaclab.assets.ArticulationCfg.actuators` before constructing the
  articulation; runtime assignment to or deletion from
  :attr:`~isaaclab.assets.Articulation.actuators` raises :class:`TypeError`.
