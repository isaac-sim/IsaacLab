Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.sensors.NewtonRaycastSensor` missing collision-only geometry by
  building the Newton scene shape BVH from both visible and colliding shapes instead of Newton's
  visible-only default. Rays now hit shapes that carry collision properties but no visual
  representation, matching the sensor's documented behavior.
