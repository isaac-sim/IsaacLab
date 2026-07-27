Added
^^^^^

* Added :mod:`isaaclab.cloner.path`, the segment-boundary-safe prim-path primitives
  :func:`~isaaclab.cloner.path.split`, :func:`~isaaclab.cloner.path.match`,
  :func:`~isaaclab.cloner.path.relativize`, :func:`~isaaclab.cloner.path.relative_to`,
  :func:`~isaaclab.cloner.path.under` and :func:`~isaaclab.cloner.path.rebase`. These
  replace :meth:`str.startswith`, :meth:`str.replace` and :meth:`str.removeprefix` on prim
  paths so a prefix test cannot match mid-segment (``".../Robot"`` no longer contains
  ``".../RobotArm"``), and they accept ``"/"`` as a root.
  :func:`~isaaclab.cloner.path.match` additionally returns the text a destination
  template's ``"{}"`` slot captured, which is how a concrete clone path names its
  environment.
* Added :mod:`isaaclab.cloner.query`, the queries over the source/clone relation a
  :class:`~isaaclab.cloner.ClonePlan` describes: :func:`~isaaclab.cloner.query.path_to_clone`
  pushes a prototype path to one environment's clone, :func:`~isaaclab.cloner.query.path_env_ids`
  returns the environments a prototype reaches, and
  :func:`~isaaclab.cloner.query.path_to_source` pulls a clone-space expression back to its
  prototype. The module docstring states the laws relating the three.
* Added an ``env_id`` argument to :func:`~isaaclab.cloner.query.path_to_source` for
  resolving a wildcard expression against one environment.
* Added :func:`isaaclab.cloner.num_spawn_variants` to the package exports.
* Added shape validation to :class:`~isaaclab.cloner.ClonePlan`, so a plan whose
  ``destinations``, ``clone_mask``, ``env_ids``, ``positions`` or ``cfg_rows`` disagree
  fails at construction instead of as an index error inside replication.

Changed
^^^^^^^

* **Breaking:** Reorganized the cloner path/plan helpers into the :mod:`isaaclab.cloner.path`
  and :mod:`isaaclab.cloner.query` submodules, accessed module-qualified (e.g.
  ``cloner.path.split`` and ``cloner.query.path_to_source``), and removed the old flat
  function names together with the ``isaaclab.cloner.cloner_utils`` module. Migrate as
  follows:

  * ``split_clone_template`` to :func:`isaaclab.cloner.path.split`
  * ``get_suffix`` to :func:`isaaclab.cloner.path.relativize`
  * ``replace_path_prefix`` to :func:`isaaclab.cloner.path.rebase`
  * ``resolve_clone_plan_source(path_expr, plan)`` to
    :func:`isaaclab.cloner.query.path_to_source` with argument order ``(plan, path_expr)``
  * ``iter_clone_plan_matches(plan, path_expr)`` to :func:`isaaclab.cloner.query.iter_sources`
  * ``ClonePlan.from_env_0(...)`` (classmethod) to the module-level function
    :func:`isaaclab.cloner.clone_plan_from_env_0`

  :func:`isaaclab.cloner.make_clone_plan`, :func:`isaaclab.cloner.make_valid_clone_combinations`,
  :func:`isaaclab.cloner.grid_transforms` and :func:`isaaclab.cloner.filter_collisions` keep
  their names and package-level import paths.

* Changed :func:`isaaclab.cloner.query.path_to_source` and
  :func:`isaaclab.cloner.query.iter_sources` to return the named tuples
  ``ResolvedSource`` and ``SourceMatch`` instead of bare tuples. Existing positional
  unpacking is unaffected.

Fixed
^^^^^

* Fixed the clone-plan queries treating environment ids as
  :attr:`~isaaclab.cloner.ClonePlan.clone_mask` column indices. A plan whose
  :attr:`~isaaclab.cloner.ClonePlan.env_ids` are not ``0..num_clones-1`` reported the wrong
  environments and indexed the wrong mask column, disagreeing with
  :func:`~isaaclab.cloner.replicate`, which formats destinations with ``env_ids[column]``.
  Environment ids outside the plan now resolve to ``None`` rather than wrapping the mask.
* Fixed :func:`isaaclab.cloner.query.path_to_source` resolving a concrete clone path in a
  heterogeneous plan to the first populated variant rather than the variant the path's own
  environment was cloned from.
* Fixed the source-side queries picking whichever plan row came first when prototypes nest,
  while the clone-side queries picked the most specific one. Both sides now resolve to the
  nearest owning row.
* Fixed :func:`isaaclab.cloner.path.split` accepting a template with more than one ``"{}"``
  slot, which left a stray slot in the suffix and broke the later ``str.format`` call.

* Fixed :func:`isaaclab.cloner.query.iter_sources` (previously ``iter_clone_plan_matches``)
  silently dropping asset variants in scenes with more than ten environments. Candidate
  rows were ranked by the length of their destination template with the row's first
  environment id substituted, so a variant starting at ``env_10`` outranked one starting at
  ``env_0`` and the latter was discarded. Ranking now uses the path suffix below the
  template, which does not depend on how many digits an environment id has. Sensors and
  ray casters resolving through a heterogeneous plan previously missed the dropped
  variant's prototype.
