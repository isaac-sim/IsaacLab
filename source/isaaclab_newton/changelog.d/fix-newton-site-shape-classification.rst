Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.sim.views.NewtonSiteFrameView` rejecting
  non-colliding Newton shape records, including MJCF sites and visual-only
  shapes, after model finalization while keeping collision shape expressions
  rejected before and after finalization.
