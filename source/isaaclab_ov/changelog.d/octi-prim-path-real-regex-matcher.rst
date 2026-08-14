Changed
^^^^^^^

* Changed prim path expressions to spell a single path segment ``[^/]`` rather than ``.``, so each
  pattern selects what it selected before now that ``.`` matches ``/`` in
  :func:`~isaaclab.sim.utils.find_matching_prims`.

Fixed
^^^^^

* Fixed the OVRTX deformable render bindings leaving the environment slot unresolved, so they
  bound against a path expression instead of the concrete per-environment mesh prims.
* Fixed physics views receiving a regular expression where the engine expects a glob. The
  conversion rewrote only ``.*`` and left a segment-safe wildcard untouched, so the view matched
  no bodies; it now goes through :func:`~isaaclab.sim.utils.path_expr_to_glob`.
