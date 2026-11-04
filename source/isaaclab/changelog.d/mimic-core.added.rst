* Added a reusable self-collision reward that counts physics frames with filtered
  robot-to-robot contact forces above a threshold.
* Added :class:`~isaaclab.utils.datasets.MotionLoader` for named NPZ reference motions,
  with local and server-hosted files, joint/body reordering and data validation.
