Added
^^^^^

* Added :func:`~isaaclab_tasks.core.velocity.mdp.rewards.feet_air_time_variance`, penalizing an
  uneven swing/stance split between a biped's feet.

Changed
^^^^^^^

* Changed ``Isaac-Velocity-Flat-Cassie`` and ``Isaac-Velocity-Rough-Cassie`` to include the new
  ``air_time_variance`` reward term, at weight ``-10.0`` and ``-5.0`` respectively. Without it the
  tasks converge on a lopsided gait, since ``feet_air_time_positive_biped`` scores the length of
  the single-stance phase without regard to which foot is swinging. Policies trained on these
  tasks will differ from ones trained before this change; set the weight to ``0.0`` to recover the
  previous reward.
