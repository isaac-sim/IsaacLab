Changed
^^^^^^^

* **Breaking:** Changed :class:`~isaaclab_newton.sim.views.NewtonSiteFrameView` to the
  frame-view two-phase lifecycle, so camera frame views are constructed with the scene
  and their sites register and clone consistently with the other sensor sites.
  Resolving frames against the finalized model is no longer supported: constructing a
  view after the Newton model is built raises ``RuntimeError``. Construct frame views
  before :meth:`~isaaclab.sim.SimulationContext.reset` instead.

Fixed
^^^^^

* Fixed the Newton multi-mesh ray caster registering glob-style patterns for tracked
  clone-plan targets; site patterns are matched as regexes, so tracked env-scoped
  targets failed site injection when the clone plan is available at construction.
