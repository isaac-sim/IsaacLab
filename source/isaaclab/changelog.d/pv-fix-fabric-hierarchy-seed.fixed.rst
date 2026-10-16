* Fixed the Fabric frame views seeding ``omni:fabric:localMatrix`` and ``omni:fabric:worldMatrix`` as identity.
  The ``usdrt`` ``Set*XformFromUsd`` helpers author the legacy OmniHydra attributes instead, so the next Fabric
  Hierarchy update moved each parent to the origin. With GPU PhysX, this moved every camera whose environment
  reset after the first reset to its env-local pose.
