Added
^^^^^

* Added :attr:`~isaaclab.cloner.CloneCfg.clone_combinations` and
  :class:`~isaaclab.scene.SelectorCfg` support for heterogeneous scenes backed
  by direct clone plans.

* Added :func:`~isaaclab.scene.scene_add` to compose spawned scene assets into
  heterogeneous clone combinations while deduplicating equivalent environment
  definitions and validating shared global assets.
* Added a direct clone-only demo that composes registered flat PhysX task
  scenes, replaces task lights with one Dome light, and reports unsupported
  configurations without constructing task environments.

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
* Fixed :func:`~isaaclab.sim.schemas.modify_articulation_root_properties` raising
  ``ModuleNotFoundError: No module named 'omni.physx'`` when spawning a fixed-base
  articulation on kitless backends (e.g. Newton). The world fixed joint for
  ``fix_root_link=True`` is now authored directly with USD when ``omni.physx`` is
  unavailable, while Kit-based backends keep using ``omni.physx`` unchanged.
