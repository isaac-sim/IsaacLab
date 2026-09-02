Changed
^^^^^^^

* Changed the GR1T2 pick-place task to grasp a primitive instead of the steering wheel under the
  ``newton_mjwarp`` preset. The wheel's rim is a torus and MuJoCo has no concave mesh-mesh
  collision, so no approximation of it is graspable: imported as a mesh the hand passes through
  the rim and closes on nothing, and collapsed to a convex hull the ring fills in and the wheel
  becomes a solid disc. The stand-in is the graspable primitive from ``Isaac-Lift-Franka``, which
  runs MJWarp as its default backend. Replace it once the asset ships a convex-decomposed rim.

* Changed the GR1T2 hand collision meshes to import as meshes rather than a single convex hull
  per link under ``newton_mjwarp``, and authored a contact material on the robot. A hull fills in
  the concavities of a curved finger, so the collider sat proud of the rendered mesh; Newton also
  resolves an omitted friction value to zero, which left the hands with no friction at all.

  The PhysX path is unchanged: it keeps the steering wheel, its original spawn and start pose,
  and authors no robot material.
