* Resolved ``--checkpoint latest`` and ``best`` in RLinf playback against the training task id recorded in
  run manifests, so recipes with a separate evaluation task id find their checkpoints.
* Resolved a relative RLinf ``--config_path`` to an absolute directory, which Hydra requires.
