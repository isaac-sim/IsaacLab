Fixed
^^^^^

* Fixed :func:`~isaaclab_rl.utils.pretrained_checkpoint.get_published_pretrained_checkpoint` reporting
  every download failure as ``A pre-trained checkpoint is currently unavailable for this task.``. A
  checkpoint the asset server does not provide is still reported that way, but the message now names the
  location that was tried, the task and backends it was derived from, and what to do instead.

Changed
^^^^^^^

* Changed :func:`~isaaclab_rl.utils.pretrained_checkpoint.get_published_pretrained_checkpoint` to raise
  ``RuntimeError`` when a published checkpoint cannot be downloaded, for instance when the
  ``.pretrained_checkpoints`` cache directory is not writable, instead of returning ``None``. The
  originating error is chained as the cause. ``None`` is now returned only when the asset server does not
  provide the checkpoint, so callers that treat ``None`` as "no checkpoint published for this task" are
  unchanged; callers that relied on ``None`` to mask local download failures must catch ``RuntimeError``.
