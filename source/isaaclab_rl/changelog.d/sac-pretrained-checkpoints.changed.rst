* Changed the pretrained-checkpoint utilities to describe a checkpoint with
  :class:`~isaaclab_rl.utils.pretrained_checkpoint.PretrainedCheckpointId`, which provides its filename, log root,
  local checkpoint, review and asset-server path. It replaces ``get_pretrained_checkpoint_filename``,
  ``get_log_root_path``, ``get_latest_job_run_path``, ``get_pretrained_checkpoint_path``,
  ``get_pretrained_checkpoint_publish_path``, ``get_published_pretrained_checkpoint_path``,
  ``has_pretrained_checkpoint_job_run``, ``has_pretrained_checkpoint_job_finished``,
  ``get_pretrained_checkpoint_review_path`` and ``get_pretrained_checkpoint_review``.
  :func:`~isaaclab_rl.utils.pretrained_checkpoint.get_published_pretrained_checkpoint` keeps its signature.
