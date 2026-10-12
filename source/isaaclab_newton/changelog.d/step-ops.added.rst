* Added :meth:`~isaaclab_newton.physics.NewtonManager.step_ops`, ``capture_graph`` and ``finish_step``, and
  ``newton_backend.step_ops``: one Newton step as operations, with eager islands such as TorchScript actuators marked,
  for callers that capture it together with their own work. ``StepGraph`` and ``StepOp`` now live in
  :mod:`isaaclab.utils.step_graph` and remain importable from ``newton_backend``.
* Camera and ray-cast queries record their work directly into an active outer capture instead of launching their own
  graphs, refitting the shared BVHs on every replay.
