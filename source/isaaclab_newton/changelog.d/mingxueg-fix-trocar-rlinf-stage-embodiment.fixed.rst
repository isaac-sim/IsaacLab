* Fixed USD assets that author ``UsdPhysics.CollisionAPI`` on an Xform for PhysX mesh merging importing into
  Newton without collision shapes. The collision settings are now moved onto the meshes beneath the Xform.
* Fixed mesh colliders with a negative ``xformOp:scale`` failing in MuJoCo-Warp. The sign is now baked into the
  mesh points so the collider scale stays positive.
