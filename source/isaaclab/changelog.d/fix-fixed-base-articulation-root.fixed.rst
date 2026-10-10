* Fixed ``fix_root_link=True`` raising ``NotImplementedError`` for an articulation rooted at the fixed joint that
  attaches it to the world when the joint's ``body0`` targets the asset prim, a fixed-base layout that UsdPhysics
  allows. It now enables that joint instead of raising or adding a second world joint. A fixed joint between two
  bodies, between a body and a static collider, or between two prims that are not rigid bodies still raises the error.
  ``fix_root_link=False`` is not supported for an articulation rooted at its fixed world joint and now logs a warning.
