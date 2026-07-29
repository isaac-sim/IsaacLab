Added
^^^^^

* Added ``--video`` and ``--video_length`` to the RSL-RL, RL-Games, SKRL, and SB3 play
  benchmark adapters, recording the rollout and populating
  :attr:`~isaaclab.benchmark.schema.PlayBundle.video_path`. The field and
  :func:`~isaaclab.benchmark.builders.build_play_bundle`'s ``video_path`` argument already
  existed, but no adapter could set them, so camera tasks were benchmarked headless.
* Added :func:`~isaaclab_rl.entrypoints.common.add_video_args`,
  :func:`~isaaclab_rl.entrypoints.common.wrap_record_video_play`, and
  :func:`~isaaclab_rl.entrypoints.common.play_video_dir`. Recording starts at the first
  step rather than on a periodic trigger, since a play run is a single bounded rollout;
  ``--video_interval`` therefore stays training-only.

Fixed
^^^^^

* Fixed :attr:`~isaaclab.benchmark.schema.TrainingBundle.checkpoint_path` being reported as
  ``None`` by the RSL-RL, RL-Games, and SKRL training benchmarks. The field is now populated
  from the checkpoint the run actually wrote, so downstream play workflows can roll out a
  freshly trained policy without reconstructing the path.
