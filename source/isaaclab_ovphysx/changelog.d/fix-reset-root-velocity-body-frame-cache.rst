Fixed
^^^^^

* Fixed :meth:`~isaaclab_ovphysx.assets.Articulation` root-velocity resets not
  invalidating the cached body-frame root velocities
  (``root_lin_vel_b`` / ``root_ang_vel_b``). Because those buffers are refreshed
  on a simulation-timestamp change, which a reset does not advance, the first
  observation after a cold ``reset()`` returned stale (uninitialized) velocities.
  For closed-loop articulations such as Cassie this surfaced as ~1e5 base-velocity
  observations at eval time, driving action saturation and a reward blowup.
