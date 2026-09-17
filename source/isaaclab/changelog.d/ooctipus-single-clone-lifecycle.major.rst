Changed
^^^^^^^

* **Breaking:** Published :class:`~isaaclab.cloner.ReplicateSession` plans on entry and dispatched
  them on exit. Empty :class:`~isaaclab.scene.InteractiveScene` configurations now author only
  ``env_0``; direct workflows must finish setup with :func:`~isaaclab.cloner.clone_plan_from_env_0`
  and :func:`~isaaclab.cloner.replicate`.
