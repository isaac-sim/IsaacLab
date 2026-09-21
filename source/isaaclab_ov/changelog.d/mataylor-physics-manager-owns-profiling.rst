Changed
^^^^^^^

* Renamed :meth:`OvPhysxManager.step` to ``OvPhysxManager._step``. The physics-step timer that used
  to bracket :meth:`~isaaclab.sim.simulation_context.SimulationContext.step` now lives in
  :meth:`~isaaclab.physics.PhysicsManager.step`, which calls each backend's ``_step`` -- backends no
  longer need to know that profiling exists. See ``source/isaaclab/changelog.d/physics-nvtx-profile-scope.rst``.
