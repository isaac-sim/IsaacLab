Added
^^^^^

* Added automatic Weights & Biases checkpoint resolution to the RSL-RL ``train`` and ``play``
  entrypoints. Passing a wandb run URL (``https://wandb.ai/<entity>/<project>/runs/<run_id>``,
  optionally with a ``?checkpoint=<iteration>`` query) or a ``wandb:<entity>/<project>/<run_id>``
  shorthand as ``--checkpoint`` downloads and loads that run's checkpoint with no extra arguments.
* Starting a new ``--logger wandb`` training run with RSL-RL now prints the
  ``wandb:<entity>/<project>/<run_id>`` shorthand for the run being created, so it can be copied
  straight into a later ``--checkpoint``.
