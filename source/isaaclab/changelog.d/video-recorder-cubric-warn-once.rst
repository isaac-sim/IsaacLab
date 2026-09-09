Fixed
^^^^^

* Fixed the :class:`~isaaclab.envs.utils.video_recorder.VideoRecorder` Kit/Newton cubric
  warning being logged on every captured frame, which flooded the terminal during
  ``--video`` runs (roughly one line per frame for the whole clip). The warned-about
  condition is fixed configuration state, so the message is now emitted once per
  recorder, matching the existing once-only behavior of the frame-capture error path.
