Added
^^^^^

* Added :mod:`isaaclab.cloner.path` and :mod:`isaaclab.cloner.query`, holding the
  segment-boundary-safe prim-path primitives and the clone-plan queries respectively.

Changed
^^^^^^^

* **Breaking:** Reorganized the cloner path/plan helpers into the :mod:`isaaclab.cloner.path`
  and :mod:`isaaclab.cloner.query` submodules, accessed module-qualified, and removed the
  ``isaaclab.cloner.cloner_utils`` module. Migrate as follows:

  * ``split_clone_template`` to :func:`isaaclab.cloner.path.split`
  * ``get_suffix`` to :func:`isaaclab.cloner.path.relativize`
  * ``replace_path_prefix`` to :func:`isaaclab.cloner.path.rebase`
  * ``resolve_clone_plan_source(path_expr, plan)`` to
    :func:`isaaclab.cloner.query.path_to_source` with argument order ``(plan, path_expr)``
  * ``iter_clone_plan_matches(plan, path_expr)`` to :func:`isaaclab.cloner.query.iter_sources`
  * ``ClonePlan.from_env_0(...)`` to :func:`isaaclab.cloner.clone_plan_from_env_0`

Fixed
^^^^^

* Fixed the clone-plan queries treating environment ids as
  :attr:`~isaaclab.cloner.ClonePlan.clone_mask` column indices, so a plan whose
  :attr:`~isaaclab.cloner.ClonePlan.env_ids` are not ``0..num_clones-1`` reported the wrong
  environments and indexed the wrong column, disagreeing with
  :func:`~isaaclab.cloner.replicate`.
* Fixed :func:`isaaclab.cloner.query.iter_sources` (previously ``iter_clone_plan_matches``)
  dropping asset variants in scenes with more than ten environments, because rows were ranked
  by their destination template with the row's first environment id substituted.
* Fixed :func:`isaaclab.cloner.query.path_to_source` resolving a concrete clone path in a
  heterogeneous plan to the first populated variant rather than the one its own environment
  was cloned from.
* Fixed the prototype-side queries picking the first matching plan row where the clone-side
  queries picked the most specific one. Both now resolve to the nearest owning row.
* Fixed :func:`isaaclab.cloner.path.split` accepting a template with more than one ``"{}"``
  slot, and the path primitives not accepting ``"/"`` as a root.
