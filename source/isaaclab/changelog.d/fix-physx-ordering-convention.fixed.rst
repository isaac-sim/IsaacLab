* **Breaking:** Fixed the ``"physx"`` convention to mean the OVPhysX order on every backend: breadth-first from
  the root PhysX elects, visiting child links by joint type, then joint prim path. Non-PhysX backends now resolve it
  from the source USD instead of a temporary Newton model, so ``"physx"`` on Newton changes for Digit joints and for
  the bodies of Go2, G1 and similar assets.
