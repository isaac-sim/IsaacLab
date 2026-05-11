Added
^^^^^

* Added explicit ``spawn_paths`` support to multi-asset spawners so scene
  planning can spawn representative heterogeneous sources directly.

Changed
^^^^^^^

* **Breaking:** Changed :class:`~isaaclab.scene.InteractiveScene` to build clone
  plans directly from asset configuration, spawn representative sources in their
  selected environments, and replicate from those sources instead of spawning and
  discovering prototypes under ``/World/template``.
* **Breaking:** Replaced ``TemplateCloneCfg`` with
  :class:`~isaaclab.cloner.CloneCfg` for clone execution settings.
* **Breaking:** Changed :func:`~isaaclab.cloner.make_clone_plan` to return a
  :class:`~isaaclab.cloner.ClonePlan` object directly.
* **Breaking:** Changed clone plan publication to use
  :meth:`~isaaclab.sim.SimulationContext.get_clone_plan` and
  :meth:`~isaaclab.sim.SimulationContext.set_clone_plan` for the single scene
  clone plan.

Removed
^^^^^^^

* **Breaking:** Removed :func:`~isaaclab.cloner.clone_from_template`. Use
  :func:`~isaaclab.cloner.make_clone_plan`,
  :func:`~isaaclab.cloner.usd_replicate`, and backend physics replication
  functions for direct cloning workflows.
