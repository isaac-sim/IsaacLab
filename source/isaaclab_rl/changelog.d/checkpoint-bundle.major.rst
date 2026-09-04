Added
^^^^^

* Added :class:`~isaaclab_rl.utils.pretrained_checkpoint.Workflow`, one object per RL library holding the
  policy-file glob, its preferred (best or final) file, the published extension and the Hydra key naming the
  experiment directory. ``WORKFLOWS`` maps workflow names to it. Its
  :meth:`~isaaclab_rl.utils.pretrained_checkpoint.Workflow.selector_args` produces the pattern arguments of
  :func:`~isaaclab_rl.entrypoints.common.resolve_checkpoint_selector`, which the play, train and benchmark
  entrypoints now use instead of spelling each library's patterns themselves.

Changed
^^^^^^^

* **Breaking:** Replaced the free functions of :mod:`isaaclab_rl.utils.pretrained_checkpoint` that took a
  ``(workflow, task_name, physics_backend, render_backend)`` tuple with the
  :class:`~isaaclab_rl.utils.pretrained_checkpoint.CheckpointBundle` class, which owns the published,
  cached and collected paths of one task variant and the checkpoints its components declare. Local
  training-run state moved to the publish script's ``CheckpointJob``.
  :func:`~isaaclab_rl.utils.pretrained_checkpoint.get_published_pretrained_checkpoint` is unchanged.
  Migration, with ``b = CheckpointBundle(workflow, task_name, physics_backend, render_backend)``:

  .. list-table::
     :header-rows: 1

     * - Removed
       - Replacement
     * - ``get_pretrained_checkpoint_filename(*t)``
       - ``b.filename()``
     * - ``get_pretrained_checkpoint_backend_names(env_cfg)``
       - ``CheckpointBundle.backend_names(env_cfg)``
     * - ``get_declared_checkpoints(env_cfg)``
       - ``CheckpointBundle.declared_companions(env_cfg)``
     * - ``get_declared_checkpoint_path(path, workflow, ckpt)``
       - ``b.published_path(ckpt)`` or ``b.collected_path(dir, ckpt)``
     * - ``get_published_pretrained_checkpoint_path(*t)``, ``get_pretrained_checkpoint_publish_path(*t)``
       - ``b.published_path()``
     * - ``get_log_root_path(*t)``, ``get_latest_job_run_path(*t)``, ``get_pretrained_checkpoint_path(*t)``,
         ``has_pretrained_checkpoint_job_run(*t)``, ``has_pretrained_checkpoint_job_finished(*t)``,
         ``get_pretrained_checkpoint_review_path(*t)``, ``get_pretrained_checkpoint_review(*t)``
       - Removed. ``scripts/tools/train_and_publish_checkpoints.py`` owns training-run state on its
         ``CheckpointJob`` (``log_root``, ``latest_run``, ``trained_path``, ``has_run``, ``has_finished``,
         ``review_path``, ``review``).
     * - ``has_pretrained_checkpoints_asset_root_dir()``
       - ``bool(isaaclab.utils.assets.NUCLEUS_ASSET_ROOT_DIR)``
     * - ``get_latest_file_or_directory(path, pattern)``
       - :func:`isaaclab.utils.io.latest_file`
     * - ``WORKFLOW_PRETRAINED_CHECKPOINT_FILENAMES``, ``WORKFLOW_PRETRAINED_CHECKPOINT_EXTENSIONS``
       - ``b.filename()``, ``WORKFLOWS[workflow].extension``
     * - ``WORKFLOW_EXPERIMENT_NAME_VARIABLE[workflow]``
       - ``WORKFLOWS[workflow].experiment_variable``
     * - ``WORKFLOW_TRAINER``, ``WORKFLOW_PLAYER``
       - Removed without replacement. Both mapped every workflow to the unified ``train.py`` / ``play.py``.

  ``WORKFLOWS`` is now a ``dict`` keyed by workflow name; iterating it and membership tests are unchanged.

* Changed ``--checkpoint best`` for skrl to prefer ``checkpoints/best_agent.pt`` when it exists, matching
  what the publish tooling has always collected.
* Changed ``train_and_publish_checkpoints.py`` to select the trained policy through the same run-manifest
  based lookup as ``--checkpoint best``, so a run must have been produced by the unified train entrypoint to
  be collected, and to derive each job's backend names and declared checkpoints from the config its preset
  selectors produce, so the published filename describes what was trained.
* Changed ``train_and_publish_checkpoints.py --publish_root`` to keep the legacy per-task sub-directory,
  so legacy checkpoints of different tasks no longer overwrite each other under a custom root.
