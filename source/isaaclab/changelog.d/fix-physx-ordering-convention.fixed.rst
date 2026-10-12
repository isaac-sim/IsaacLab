* Fixed the ``"physx"`` convention to mean the OVPhysX order on every backend: child links are visited by joint
  type, then joint prim path. Non-PhysX backends now resolve it from the source USD instead of a temporary Newton
  model, which also corrects Digit joints and the bodies of Go2, G1 and similar assets on Newton.
