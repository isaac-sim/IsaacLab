Added
^^^^^

* Added :attr:`~isaaclab.cloner.ClonePlan.global_paths` to identify scene assets shared by every environment
  without representing them as replication rows.

Changed
^^^^^^^

* Changed :func:`~isaaclab.cloner.make_clone_plan`, :func:`~isaaclab.cloner.clone_plan_from_env_0`, and
  :class:`~isaaclab.cloner.ReplicateSession` to accept explicit ``global_paths`` tuples.
