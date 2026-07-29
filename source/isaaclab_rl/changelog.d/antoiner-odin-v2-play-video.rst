Added
^^^^^

* Added :func:`~isaaclab_rl.entrypoints.common.add_video_args`,
  :func:`~isaaclab_rl.entrypoints.common.wrap_record_video_play`, and
  :func:`~isaaclab_rl.entrypoints.common.play_video_dir` so play workflows can record a
  rollout video. ``add_common_train_args`` now delegates its video arguments to the
  shared helper.
