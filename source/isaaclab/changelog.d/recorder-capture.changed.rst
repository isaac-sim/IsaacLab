* Replaced per-world recorder episode containers with batched field allocations and
  independent insertion indices, reusing storage across resets and growing without dropping samples.
* Added :meth:`~isaaclab.managers.RecorderManager.get_term` and the term capability
  :attr:`~isaaclab.managers.RecorderTerm.requires_post_step_observations` so physical-state
  capture can avoid extra observation computation and noise draws.
* **Breaking:** Changed :meth:`~isaaclab.managers.RecorderManager.get_episode` to return an owned
  snapshot with tensor leaves. To update live recording, use ``add_to_episodes`` and
  ``set_success_to_episodes``; do not mutate the snapshot. Set ``buffer_initial_capacity``
  to control the initial allocation per field.
