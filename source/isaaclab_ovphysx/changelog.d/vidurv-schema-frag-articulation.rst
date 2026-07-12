Fixed
^^^^^

* Fixed ``fix_root_link=True`` for OVPhysX fragment-based articulation spawns by relocating the
  articulation root to the parser-required parent, including when an existing disabled world fixed
  joint is re-enabled.
