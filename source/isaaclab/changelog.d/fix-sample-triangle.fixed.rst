* Fixed :func:`~isaaclab.utils.math.sample_triangle` sampling a V-shaped distribution with the least density at the
  midpoint and the most at the bounds. It now samples the documented symmetric triangular distribution, with its mode
  at the midpoint of ``lower`` and ``upper``.
