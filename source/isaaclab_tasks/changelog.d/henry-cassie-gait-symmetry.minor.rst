Added
^^^^^

* Added :func:`~isaaclab_tasks.core.velocity.mdp.rewards.feet_air_time_variance`, penalizing an
  uneven swing/stance split between a biped's feet.

Changed
^^^^^^^

* Changed ``Isaac-Velocity-Flat-Cassie``, ``Isaac-Velocity-Rough-Cassie`` and
  ``Isaac-Velocity-Flat-H1`` to include the new ``air_time_variance`` reward term, which fixes the
  tilted gait these tasks otherwise converge on. Policies trained on them will differ from ones
  trained before this change; set the weight to ``0.0`` to recover the previous reward.
