Fixed
^^^^^

* Fixed collision shapes being rendered on top of their visual geometry after the Newton bump, by
  requesting Newton's ``hide_collision_shapes`` import behavior in the cloner so colliders are only
  shown for bodies and static parents that have no separate visual shape.
