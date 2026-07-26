Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.sensors.NewtonRaycastSensor` missing collision-only geometry. A scene
  containing a ray-cast sensor now rebuilds the Newton shape BVH from both visible and colliding
  shapes, so rays hit shapes that carry collision properties but no visual representation. Scenes
  without a ray-cast sensor keep Newton's visible-only BVH, leaving camera renders unchanged.
