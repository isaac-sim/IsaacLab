Changed
^^^^^^^

* Changed ``train_multigpu`` to restrict console output to local rank 0 on each node by default, since
  every rank otherwise repeats the same startup, warning, and model-summary output once per GPU. Pass
  ``--log_all_ranks`` to restore output from every rank, or ``--tee 3 --log_dir <dir>`` to keep
  per-rank logs on disk while the console stays clean. Does not apply to skrl with
  ``--ml_framework jax``, whose launcher does not support rank filtering.

Fixed
^^^^^

* Fixed distributed training failures on non-zero ranks being reported as a bare exit code with no
  traceback. The training entry point now records the failing rank's traceback so ``torchrun``
  reports it as the root cause.
