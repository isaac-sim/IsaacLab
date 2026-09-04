Fixed
^^^^^

* Fixed ``play.py`` (all RL library backends) stopping the rollout after
  ``video_recorders[0].video_length`` steps instead of ``video_length + step_offset`` steps
  when ``--video`` is passed without an explicit ``--video_length``, which silently truncated
  clips recorded with a nonzero :attr:`~isaaclab.envs.utils.video_recorder_cfg.VideoRecorderCfg.step_offset`.
