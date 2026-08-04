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

* Fixed ``--video`` on kitless OvPhysX selecting the Kit capture backend. The OvPhysX
  physics manager is named ``OvPhysxManager``, so a substring test for ``physx`` routed it
  to a recorder that imports ``omni.replicator`` lazily, and the run failed on the first
  rendered frame after simulation startup and policy load. OvPhysX has no capture backend,
  so it now raises at backend selection with a message naming the alternatives.
* Fixed :attr:`~isaaclab.benchmark.schema.TrainingBundle.checkpoint_path` being reported as
  ``None`` by the RSL-RL, RL-Games, and SKRL training benchmarks. The field is now populated
  from the checkpoint the run actually wrote, so downstream play workflows can roll out a
  freshly trained policy without reconstructing the path. The search matches every library's
  naming, including SKRL's ``agent_<tag>.pt`` and ``best_agent.pt``.
