Fixed
^^^^^

* Fixed :func:`~isaaclab.sim.utils.resolve_matching_prims_from_source` returning duplicate
  prims when a path expression matched overlapping ancestor subtrees. Descendant matches
  were deduplicated by full prim path before validating the match count, preserving their
  first-match order and destination expressions for all asset and sensor callers.
