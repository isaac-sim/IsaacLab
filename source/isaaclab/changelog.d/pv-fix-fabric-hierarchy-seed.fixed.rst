* Fixed the Fabric frame views seeding ``omni:fabric:localMatrix`` and ``omni:fabric:worldMatrix`` as identity,
  which moved every camera whose environment reset after the first reset to its env-local pose with GPU PhysX.
