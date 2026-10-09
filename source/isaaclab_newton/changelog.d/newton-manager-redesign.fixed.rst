* Fixed Newton actuator telemetry and actuator-history resets when worlds hold different DOF layouts. The actuator
  adapter used a uniform per-environment DOF stride, which undersized its computed-effort buffer.
* Fixed step callbacks accumulating duplicates across hard resets; they now share the lifetime of the backend they
  bind to.
* Fixed authored-state invalidation of global articulations, which flagged world 0 for a solver reset.
* Fixed particle forces applying only to the first solver substep; the step stages them like body forces.
* Fixed articulations running Isaac Lab actuator models while the Newton manager ran the decimation loop; they now
  require the environment to run the loop.
* Fixed IMU, PVA, and frame-transformer sensors injecting duplicate sites into the retained builder on every hard
  reset. Site requests now persist until :meth:`~isaaclab_newton.physics.NewtonManager.close`, a resolved site is
  never injected again, and sensors no longer re-register on ``STOP``.
* Fixed every CUDA graph capture running a full Python garbage collection afterwards, which added about 0.25 s per
  capture on large scenes (about 1.7 s at the first step of the rough-terrain ANYmal-D task).
