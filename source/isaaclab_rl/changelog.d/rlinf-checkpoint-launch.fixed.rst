* Fixed RLinf playback ignoring ``--checkpoint`` and training resume selecting a directory below
  ``global_step_<N>``. Checkpoint files, directories containing one ``full_weights.pt``, and the existing
  ``latest``/``best`` selectors remained supported.
* Resolved RLinf model and checkpoint paths before launching Ray workers and merged actor-model settings
  into rollout settings while preserving explicit rollout overrides.
