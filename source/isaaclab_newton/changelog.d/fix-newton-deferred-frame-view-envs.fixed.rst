* Fixed :class:`~isaaclab_newton.sim.views.NewtonSiteFrameView` instances created before the Newton model exists
  not covering the environments they select. A static frame created after the scene was cloned collapsed to one
  frame without an environment offset, and a view selecting some environments got a frame in every environment.
  Such views now resolve against the finalized model, as views created after reset do.
