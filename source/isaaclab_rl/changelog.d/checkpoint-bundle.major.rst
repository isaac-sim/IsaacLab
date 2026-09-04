Changed
^^^^^^^

* **Breaking:** Replaced the free functions of :mod:`isaaclab_rl.utils.pretrained_checkpoint` that took a
  ``(workflow, task_name, physics_backend, render_backend)`` tuple with the
  :class:`~isaaclab_rl.utils.pretrained_checkpoint.CheckpointBundle` class, which owns the published,
  cached, trained and reviewed paths of one task variant and the checkpoints its components declare.
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
       - ``CheckpointBundle.from_env_cfg(workflow, task_name, env_cfg).checkpoints``
     * - ``get_declared_checkpoint_path(path, workflow, ckpt)``
       - ``b.published_path(ckpt)``, ``b.collected_path(dir, ckpt)`` or ``b.trained_path(ckpt)``
     * - ``get_published_pretrained_checkpoint_path(*t)``, ``get_pretrained_checkpoint_publish_path(*t)``
       - ``b.published_path()``
     * - ``get_log_root_path(*t)``, ``get_latest_job_run_path(*t)``
       - ``b.log_root``, ``b.latest_run``
     * - ``get_pretrained_checkpoint_path(*t)``
       - ``b.trained_path()``
     * - ``has_pretrained_checkpoint_job_run(*t)``, ``has_pretrained_checkpoint_job_finished(*t)``
       - ``b.has_run``, ``b.has_finished``
     * - ``get_pretrained_checkpoint_review_path(*t)``, ``get_pretrained_checkpoint_review(*t)``
       - ``b.review_path``, ``b.review``
     * - ``has_pretrained_checkpoints_asset_root_dir()``
       - ``bool(isaaclab.utils.assets.NUCLEUS_ASSET_ROOT_DIR)``
     * - ``get_latest_file_or_directory(path, pattern)``
       - :func:`isaaclab.utils.io.latest_file`
     * - ``WORKFLOW_PRETRAINED_CHECKPOINT_FILENAMES``, ``WORKFLOW_PRETRAINED_CHECKPOINT_EXTENSIONS``
       - ``b.filename()``, ``WORKFLOW_POLICY[workflow].extension``
     * - ``WORKFLOW_TRAINER``, ``WORKFLOW_PLAYER``
       - Removed without replacement. Both mapped every workflow to the unified ``train.py`` / ``play.py``.

* Changed ``train_and_publish_checkpoints.py --publish_root`` to keep the legacy per-task sub-directory,
  so legacy checkpoints of different tasks no longer overwrite each other under a custom root.
