Changed
^^^^^^^

* **Breaking:** Changed :func:`~isaaclab.cloner.clone_plan_from_env_0` to accept a
  :class:`~isaaclab.cloner.CloneCfg` and a flat asset-cfg sequence, publish the plan before asset
  construction, and require :func:`~isaaclab.cloner.replicate` to dispatch that active plan.
  ``REPLICATION_QUEUE`` and ``queue_replication`` were removed; pass the declared clone cfg and
  complete flat asset/sensor manifest. Use :class:`~isaaclab.cloner.ReplicateSession` for heterogeneous layouts.
* Changed Direct environments to construct ``cfg.scene.class_type(cfg.scene)`` before the optional
  ``_setup_scene`` hook. Declare normal Direct-workflow assets and sensors on ``cfg.scene`` so
  :class:`~isaaclab.scene.InteractiveScene` owns their construction and clone lifecycle.
