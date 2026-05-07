Added
^^^^^

* Added :class:`~isaaclab.cloner.ClonePlan` as the single source of truth for
  clone sources, destination templates, and the source-to-environment mask.
* Added :meth:`~isaaclab.sim.SimulationContext.get_clone_plan` and
  :meth:`~isaaclab.sim.SimulationContext.set_clone_plan` for publishing the
  scene clone plan.
* Added explicit ``spawn_paths`` support to multi-asset spawners so scene
  planning can spawn representative heterogeneous sources directly.

Changed
^^^^^^^

* **Breaking:** :class:`~isaaclab.scene.InteractiveScene` now builds clone plans
  directly from asset configuration, spawns representative sources in their
  selected environments, and replicates from those sources. This removes the old
  template-spawn and prototype-discovery round trip.
* **Breaking:** Replaced ``TemplateCloneCfg`` with
  :class:`~isaaclab.cloner.CloneCfg` for clone execution settings.
* Changed :func:`~isaaclab.cloner.make_clone_plan` to return a
  :class:`~isaaclab.cloner.ClonePlan` object directly.

Removed
^^^^^^^

* **Breaking:** Removed :func:`~isaaclab.cloner.clone_from_template`. Use
  :func:`~isaaclab.cloner.make_clone_plan`,
  :func:`~isaaclab.cloner.usd_replicate`, and backend physics replication
  functions for direct cloning workflows.
