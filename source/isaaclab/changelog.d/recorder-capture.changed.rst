* Replaced per-world recorder episode containers with batched field allocations and
  independent insertion indices, reusing storage across resets and growing without dropping samples.
* Added :meth:`~isaaclab.managers.RecorderManager.get_term` for direct access to captured batches.
* Preserve slices through recorder reset callbacks; terms accept slices or indices directly.
* **Breaking:** Removed the implicit observation computation before RL post-step recorder callbacks.
  Physical capture no longer consumes extra observation noise. Custom terms needing observations
  of the completed transition must call ``env.observation_manager.compute_group(name, update_history=False)``
  explicitly; ``env.obs_buf`` still contains the preceding policy input. Use a noiseless group without
  history or delay for current measurements that do not consume policy noise draws.
* **Breaking:** Changed :meth:`~isaaclab.managers.RecorderManager.get_episode` to return an owned
  snapshot with tensor leaves. To update live recording, use ``add_to_episodes`` and
  ``set_success_to_episodes``; do not mutate the snapshot. Set ``buffer_initial_capacity``
  to control the initial allocation per field.
* **Breaking:** Removed ``EpisodeData.add()`` and ``pre_export()``. Collection now belongs to
  ``RecorderManager.add_to_episodes()``; assign nested tensors with leading sample dimensions
  to ``EpisodeData.data`` when constructing an episode directly. Playback and HDF5 loading remain supported.
* **Breaking:** Removed the unused ``RecorderTerm.close(file_path)`` hook. Dataset handlers own
  file finalization; applications must explicitly close any additional resources they own.

* Documented independent capture, retention and export choices, with task-selected schemas and
  consumer-owned processing. Core recorder storage remains independent of task and learning packages.
