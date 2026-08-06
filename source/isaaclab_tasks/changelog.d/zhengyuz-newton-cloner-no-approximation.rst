Changed
^^^^^^^

* Changed the Lift multi-asset object and the Reorient dex cube to author
  ``convexHull`` on their collision properties. Both spawn tessellated mesh
  primitives with no approximation authored, which the Newton cloner used to hull
  implicitly; authoring it keeps their collision shapes unchanged now that the
  cloner leaves USD-authored approximations alone.
