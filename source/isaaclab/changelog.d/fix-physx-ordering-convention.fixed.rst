* **Breaking:** Fixed the ``"physx"`` convention to mean the OVPhysX order on every backend: breadth-first from
  the root PhysX elects, visiting child links by joint type, then joint prim path. ``"physx"`` on Newton changes for
  the joints of Digit and for the bodies of Go2, G1, Galbot and the Franka mimic asset, and now resolves when
  ``articulation_root_prim_path`` is a pattern. It roots at the ``ArticulationRootAPI`` body before any world joint,
  and raises instead of guessing when a floating articulation rooted on a container prim has a tied graph center, as
  Humanoid-AMP does, nested bodies or loop-closing joints.
