Added
^^^^^

* Added :attr:`~isaaclab.cloner.CloneCfg.clone_combinations` support for
  heterogeneous scenes backed by direct clone plans.
* Added :func:`~isaaclab.scene.add` to fold environment-scoped scene
  assets into heterogeneous clone combinations while deduplicating equivalent
  environment definitions.
* Added a direct clone-only demo that composes registered flat PhysX task
  scenes, excludes task scenes whose floor is not at level 0, and replaces
  task lights and floors with one Dome light and one shared ground plane,
  without constructing task environments.

Changed
^^^^^^^

* Changed the :attr:`~isaaclab.cloner.CloneCfg.clone_strategy` default from
  :func:`~isaaclab.cloner.random` to :func:`~isaaclab.cloner.sequential`,
  matching :func:`~isaaclab.cloner.make_clone_plan` and
  :class:`~isaaclab.cloner.ReplicateSession`. Set
  ``clone_cfg.clone_strategy = random`` explicitly to keep random
  prototype-to-environment assignment.

Fixed
^^^^^

* Fixed :func:`~isaaclab.cloner.resolve_clone_plan_source` raising
  ``NotImplementedError`` for assets cloned into only a subset of envs, which
  blocked heterogeneous scenes where a robot type is present in just one task
  group. Partial-env coverage now resolves to a destination glob spanning only
  the envs that received the asset.
* Fixed static :class:`~isaaclab.assets.AssetBaseCfg` assets (e.g. tables) being
  spawned only into their source env in heterogeneous scenes. They are now queued
  for USD replication so the clone plan spreads them to every env of their group.
* Fixed USD replication double-applying per-env grid origins to assets cloned into
  nested (sub-env) destinations, which offset static assets to the wrong place.
  Env-origin transforms are now authored only on env-root prims; nested assets keep
  their intra-env transform and inherit the origin from their env parent.
