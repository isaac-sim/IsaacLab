Added
^^^^^

* Added backend-neutral collision groups with whole-path regular-expression selectors and per-group
  deny-list or allow-list relationships, realized by the active physics manager during cloning.

Changed
^^^^^^^

* Changed replicated-environment collision isolation to a cloner-owned request passed through
  :class:`~isaaclab.cloner.ReplicateSession` and :func:`~isaaclab.cloner.replicate`; direct workflows
  no longer need a separate post-cloning filtering call.

Deprecated
^^^^^^^^^^

* Deprecated :meth:`isaaclab.scene.InteractiveScene.filter_collisions` in favor of passing
  ``isolate_environments`` to the cloning lifecycle. Legacy filtering calls after the physics-manager
  assembly barrier are rejected.
