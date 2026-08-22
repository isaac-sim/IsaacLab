Added
^^^^^

* Added :attr:`~isaaclab.cloner.ClonePlan.global_paths` to identify scene assets shared by every environment
  without representing them as replication rows.

Changed
^^^^^^^

* **Breaking:** Changed :func:`~isaaclab.cloner.clone_plan_from_env_0` to require ``global_paths``. Pass every
  shared asset root, or ``global_paths=()`` when the hand-built scene has none.
