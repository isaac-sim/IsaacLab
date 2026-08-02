Added
^^^^^

* Added :attr:`~isaaclab.cloner.CloneCfg.clone_template` for the replicated environment prim path,
  with ``{}`` marking the environment index. It replaces ``CloneCfg.clone_regex``, whose value is
  now ``clone_template.format("[^/]+")``.
* Added an ``env_template`` argument to :func:`~isaaclab.cloner.make_clone_plan` and
  :class:`~isaaclab.cloner.ReplicateSession`.
* Added :func:`~isaaclab.sim.utils.path_expr_to_glob` and
  :func:`~isaaclab.sim.utils.split_path_expr`, for converting a prim path expression to the glob
  the physics engines accept and for splitting one without cutting a character class in half.

Changed
^^^^^^^

* **Breaking:** Changed :func:`~isaaclab.sim.utils.find_matching_prims` to match the whole prim
  path as a plain regular expression instead of one token per path segment. ``.`` now matches
  ``/``, so ``/World/Robot/.*`` selects descendants at any depth; use ``[^/]+`` for a single
  segment.
* Changed :func:`~isaaclab.sim.utils.find_first_matching_prim` to delegate to
  :func:`~isaaclab.sim.utils.find_matching_prims`, so both read an expression the same way.
* Changed the environment namespace to spell its slot ``[^/]+`` rather than ``.*``, so
  ``{ENV_REGEX_NS}/Robot`` no longer also selects a ``Robot`` nested deeper under an environment.
* Changed :func:`~isaaclab.cloner.path.match` to accept a character class in the clone slot, so a
  segment-safe namespace resolves against a destination template.

* Changed prim path expressions throughout the repository to spell a single path segment
  ``[^/]`` rather than ``.``, so each pattern selects what it selected before now that ``.``
  matches ``/``.

Removed
^^^^^^^

* Removed the legacy glob-wildcard rewrite from prim path expressions. A bare ``*`` is a regular
  expression quantifier and is no longer rewritten to ``.*``; the rewrite could not tell a glob
  star from a quantifier and corrupted ``[^/]*`` into ``[^/].*``. Patterns relying on ``*`` as a
  standalone wildcard should spell it ``.*`` (any depth) or ``[^/]*`` (one path segment).

Fixed
^^^^^

* Fixed :func:`~isaaclab.cloner.make_clone_plan` raising ``IndexError`` for a prim path holding
  more than one wildcard, and ignoring a non-default environment namespace.
* Fixed :class:`~isaaclab.sensors.MultiMeshRayCaster` expanding ``{ENV_REGEX_NS}`` with a
  hardcoded namespace instead of the shared default.
* Fixed callers that split a prim path expression on ``/`` cutting a ``[^/]`` character class in
  half, which raised ``re.error: unterminated character set`` or produced a truncated body name.
* Fixed :func:`~isaaclab.sim.spawn_multi_asset` rejecting an index slot spelled ``[^/]*``; the
  slot is now any segment wildcard rather than a literal ``.*``.
* Fixed callers that substituted a concrete environment index into a path expression by matching
  one spelling of the environment slot, so a namespace written with a different quantifier was
  left unresolved: the visualizer camera view, and the deformable render bindings.
* Fixed :func:`~isaaclab.cloner.query.path_to_source` reporting its destination as a glob, which
  matched nothing when a caller used it as the path expression its name promises.
