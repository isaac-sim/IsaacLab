* Fixed ``fix_root_link=True`` raising ``NotImplementedError`` for an articulation rooted at the fixed joint that
  attaches it to the world when the joint's ``body0`` targets the asset prim, the fixed-base layout that UsdPhysics
  recommends. It now enables that joint instead of raising or adding a second world joint. A fixed joint between two
  bodies, or between a body and a static collider, still raises the error.
