Added
^^^^^

* Overrode :meth:`provides_implicit_damping` on :class:`NewtonManager` to return ``False`` (its
  symplectic integrator has no implicit damping) and :meth:`provides_temporal_camera_data` on
  :class:`NewtonWarpRenderer` to return ``False`` (the rasterizer accumulates no temporal data),
  so camera tasks can auto-enable frame stacking for the Newton combos that need it.
