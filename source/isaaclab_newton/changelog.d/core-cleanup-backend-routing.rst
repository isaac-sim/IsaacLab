Changed
^^^^^^^

* Changed the ``set_coms_index`` and ``set_coms_mask`` methods of the Newton
  :class:`~isaaclab_newton.assets.Articulation`, :class:`~isaaclab_newton.assets.RigidObject`, and
  :class:`~isaaclab_newton.assets.RigidObjectCollection` to also accept center of mass poses (trailing dimension
  of 7 or ``wp.transformf``), like the other backends. The orientation is ignored.

Fixed
^^^^^

* Fixed mass, center of mass, and inertia changes on the Featherstone solver being dropped silently. The
  solver does not apply them after it is constructed; the Newton manager now logs a warning the first time
  such a change is made.
