Added
^^^^^

* Added the clone-plan query operations :func:`isaaclab.cloner.query.source_idx`,
  :func:`isaaclab.cloner.query.path_env_ids`, :func:`isaaclab.cloner.query.path_maps_to_env`,
  :func:`isaaclab.cloner.query.path_to_clone`, and :func:`isaaclab.cloner.query.path_to_clones`,
  which push a source-space prim path forward to one environment's clone or over its whole
  environment fiber.
* Added the segment-boundary-safe path primitives :func:`isaaclab.cloner.path.under` and
  :func:`isaaclab.cloner.path.relative_to`, which replace :meth:`str.startswith` and
  :meth:`str.removeprefix` on prim paths so a prefix test cannot match mid-segment
  (``".../Robot"`` no longer contains ``".../RobotArm"``).

Changed
^^^^^^^

* **Breaking:** Reorganized the cloner path/plan helpers into the ``isaaclab.cloner.path`` and
  ``isaaclab.cloner.query`` submodules (accessed module-qualified, e.g. ``cloner.path.split`` and
  ``cloner.query.path_to_source``) and removed the old flat function names and the
  ``isaaclab.cloner.cloner_utils`` module. Migrate as follows:

  * ``split_clone_template`` to :func:`isaaclab.cloner.path.split`
  * ``get_suffix`` to :func:`isaaclab.cloner.path.relativize`
  * ``replace_path_prefix`` to :func:`isaaclab.cloner.path.rebase`
  * ``resolve_clone_plan_source(path_expr, plan)`` to :func:`isaaclab.cloner.query.path_to_source` with
    argument order ``(plan, path_expr)``
  * ``iter_clone_plan_matches(plan, path_expr)`` to :func:`isaaclab.cloner.query.iter_sources`
  * ``ClonePlan.from_env_0(...)`` (classmethod) to the module-level function
    :func:`isaaclab.cloner.clone_plan_from_env_0` (call as ``cloner.clone_plan_from_env_0(...)``)

  :func:`isaaclab.cloner.make_clone_plan`, :func:`isaaclab.cloner.make_valid_clone_combinations`,
  :func:`isaaclab.cloner.grid_transforms`, and :func:`isaaclab.cloner.filter_collisions` keep their
  names and package-level import paths.
