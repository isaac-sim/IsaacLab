Changed
^^^^^^^

* **Breaking:** Changed :class:`~isaaclab_newton.sim.views.NewtonSiteFrameView` to the
  frame-view two-phase lifecycle, so camera frame views are constructed with the scene
  and their sites register and clone consistently with the other sensor sites.
  Resolving frames against the finalized model is no longer supported: constructing a
  view after the Newton model is built raises ``RuntimeError``. Construct frame views
  before :meth:`~isaaclab.sim.SimulationContext.reset` instead.
* **Breaking:** Changed :meth:`~isaaclab_newton.physics.NewtonManager.cl_register_site`
  to reject registrations that arrive after the scene builder exists. Sites are
  injected into the scene builders and cloned with the scene, so registering one after
  replication silently skipped cloning and could recycle another sensor's site label.
  Construct sensors and frame views inside the scene (or, without replication, before
  the simulation resets) instead.
* Changed :class:`~isaaclab_newton.sim.views.NewtonSiteFrameView` to register the
  source body path of each matched clone-plan row and merge the resulting sites per
  environment, so heterogeneously spawned assets resolve each environment's own
  variant frame.

Fixed
^^^^^

* Fixed the Newton multi-mesh ray caster registering glob-style patterns for tracked
  clone-plan targets; site patterns are matched as regexes, so tracked env-scoped
  targets failed site injection when the clone plan is available at construction.
* Fixed frame views on heterogeneously spawned assets
  (:class:`~isaaclab.sim.spawners.MultiAssetSpawnerCfg` and
  :class:`~isaaclab.sim.spawners.MultiUsdFileCfg`) duplicating frames across source
  variants; each environment now resolves exactly one frame from its own variant.
* Fixed site registration on bodies outside the cloned environments (e.g. a shared
  table or fixture) failing during replication; such sites are now injected into the
  main builder once, like global sites.
