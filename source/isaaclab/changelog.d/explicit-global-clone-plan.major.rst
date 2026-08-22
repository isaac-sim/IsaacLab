Added
^^^^^

* Added :attr:`~isaaclab.cloner.ClonePlan.global_paths` to identify scene assets shared by every environment
  without representing them as replication rows.

Changed
^^^^^^^

* **Breaking:** Changed :func:`~isaaclab.cloner.make_clone_plan`,
  :func:`~isaaclab.cloner.clone_plan_from_env_0`, and :class:`~isaaclab.cloner.ReplicateSession` to require
  ``global_paths``. Pass every shared asset root, or ``global_paths=()`` when the scene has none.
