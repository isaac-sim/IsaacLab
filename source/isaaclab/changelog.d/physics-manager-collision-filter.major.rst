Added
^^^^^

* Added backend-neutral collision groups with whole-path regular-expression selectors and per-group
  deny-list or allow-list relationships, realized by the active physics manager during cloning.

Changed
^^^^^^^

* Changed replicated-environment collision isolation to a cloner-owned request passed through
  :class:`~isaaclab.cloner.ReplicateSession` and :func:`~isaaclab.cloner.replicate`; direct workflows
  no longer need a separate post-cloning filtering call.
* Changed post-barrier collision filtering to reject legacy mutations. Existing direct workflows must pass
  ``isolate_environments`` to :func:`~isaaclab.cloner.replicate` instead of calling
  :meth:`isaaclab.scene.InteractiveScene.filter_collisions` afterward.

Deprecated
^^^^^^^^^^

* Deprecated :meth:`isaaclab.scene.InteractiveScene.filter_collisions` in favor of passing
  ``isolate_environments`` to the cloning lifecycle before the physics-manager assembly barrier.
