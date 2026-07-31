Added
^^^^^

* Added ``prim_path_regex`` to frame-transformer source and target configurations
  for matching only the prims selected by the path expression.
* Added normalized regex parsing to
  :func:`~isaaclab.sim.resolve_matching_prims_from_source`.

Deprecated
^^^^^^^^^^

* Deprecated frame-transformer ``prim_path`` fields. Existing configurations keep
  their recursive descendant lookup behavior.
