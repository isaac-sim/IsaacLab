* Fixed the Fabric frame view seed that wrote ``omni:fabric:localMatrix`` and ``omni:fabric:worldMatrix`` as
  identity. With GPU PhysX, every camera whose environment reset after the first reset moved to its env-local pose.
