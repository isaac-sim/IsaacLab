* **Breaking:** Changed Isaac Sim PhysX articulations to reorder their joint prims at spawn and to use the
  ``"physx"`` order when no ordering is configured, so joint and body order matches OVPhysX. Assets whose sibling
  joints are not authored in path order, such as Shadow Hand, Digit and Humanoid, change order. To load a policy
  trained on Isaac Sim PhysX before this change, set ``joint_ordering="isaacsim_physx"`` and
  ``body_ordering="isaacsim_physx"`` on the articulation configuration.
