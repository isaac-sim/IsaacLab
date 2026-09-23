Fixed
^^^^^

* Fixed :class:`~isaaclab.envs.utils.video_recorder_cfg.VideoRecorderCfg` accepting clip schedules the
  recorder cannot honor. Environment config validation now rejects a non-positive ``video_length`` or
  ``frame_stride`` and a negative ``video_interval`` or ``step_offset`` instead of silently writing
  one-frame clips, dividing by zero, or recording nothing.
