Added
^^^^^

* Added :func:`~isaaclab.sim.usd_export.export_articulation_to_usd` and
  :func:`~isaaclab.sim.usd_export.write_articulation_state_to_stage` to write a running
  articulation's simulated state back onto the prims it was spawned from. Properties overridden
  after the stage is parsed -- drive gains, masses, armature, joint friction and limits -- live only
  in the physics backend's buffers, so saving the stage of a running scene previously emitted the
  spawn-time values. Backends supply their own prim paths through
  :class:`~isaaclab.sim.usd_export.ArticulationPrimPaths`. The export authors onto a flattened
  snapshot of the stage, so the running simulation is never edited; ``write_articulation_state_to_stage``
  takes the target ``stage`` explicitly and defaults to the live one. Joint friction is written as the
  per-axis static, dynamic and viscous ``PhysxJointAxisAPI`` model the runtimes simulate, with
  angular quantities in the stage's degree convention.
