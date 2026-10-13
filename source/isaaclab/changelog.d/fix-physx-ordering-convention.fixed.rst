* **Breaking:** Fixed the ``"physx"`` convention to mean the OVPhysX order on every backend: breadth-first from
  the root PhysX elects, visiting child links by joint type, then joint prim path. ``"physx"`` on Newton changes for
  the joints of Digit and Humanoid-AMP and for the bodies of Go2, G1, Galbot and the Franka mimic asset, and now
  resolves when ``articulation_root_prim_path`` is a pattern.
