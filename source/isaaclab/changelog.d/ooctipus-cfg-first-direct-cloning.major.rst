Changed
^^^^^^^

* **Breaking:** Changed :func:`~isaaclab.cloner.clone_plan_from_env_0` to accept a
  :class:`~isaaclab.cloner.CloneCfg` and a flat asset-cfg sequence, and to publish the plan before
  asset construction. Pass the declared clone cfg and every asset or sensor cfg. Use
  :func:`~isaaclab.cloner.make_clone_plan` for heterogeneous layouts.
* **Breaking:** Removed ``REPLICATION_QUEUE`` and ``queue_replication``. Direct workflows now pass
  their complete flat construction manifest to :func:`~isaaclab.cloner.clone_plan_from_env_0`.
