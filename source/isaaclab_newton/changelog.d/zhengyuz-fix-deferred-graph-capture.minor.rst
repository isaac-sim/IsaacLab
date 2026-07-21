Changed
^^^^^^^

* Changed :class:`~isaaclab_newton.physics.NewtonManager` to step through a
  single program with per-stage toggles instead of two parallel step routines.
  The private ``_simulate_full`` and ``_simulate_physics_only`` methods were
  removed; subclasses that overrode them should override ``_simulate``
  instead.
* Changed :meth:`~isaaclab_newton.physics.NewtonManager.handles_decimation` to
  return ``True`` whenever the Newton actuator path is active, including mixed
  scenes with non-graph-safe actuators (previously only when every actuator
  was CUDA-graph-safe). Environments now fold the decimation loop into one
  :meth:`~isaaclab_newton.physics.NewtonManager.step` call for such scenes;
  callers that stepped mixed scenes one sub-step at a time should gate on
  ``handles_decimation()`` as the environments do.
* Changed CUDA graph capture in
  :class:`~isaaclab_newton.physics.NewtonManager` to a single deferral policy:
  decision sites (solver initialization, decimation changes, hard resets) only
  invalidate the graph, and the first
  :meth:`~isaaclab_newton.physics.NewtonManager.step` afterwards captures it,
  with the capture warmup counting as that step's physics advance. Exactly one
  capture occurs per invalidation.

Fixed
^^^^^

* Fixed the initial CUDA graph capture being skipped instead of deferred when
  the Newton actuator fast path is active (``use_newton_actuators=True``):
  callers that never set a decimation — plain
  :class:`~isaaclab.sim.SimulationContext` loops — silently ran eager forever
  despite requesting ``use_cuda_graph=True``.
