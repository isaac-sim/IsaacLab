Fixed
^^^^^

* Fixed :attr:`~isaaclab_newton.assets.ArticulationData.body_com_vel_w` (and every derived
  body-velocity property, e.g. :attr:`body_lin_vel_w`) reading zeros for fixed-base
  articulations. The fixed-base fallback in the data buffers zeroed the body velocity
  binding together with the genuinely unavailable root velocity, silently disabling all
  body-velocity observations for fixed-base robots.
