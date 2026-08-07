Added
^^^^^

* Added :func:`~isaaclab_rl.entrypoints.run_train_multigpu_cli`, which moves the multi-GPU launcher
  into the package alongside the train and play entry points.
  ``scripts/reinforcement_learning/train_multigpu.py`` is now a shim over it.

Fixed
^^^^^

* Fixed the multi-GPU launcher leaving worker processes behind on Ctrl-C. It ran torchrun in its own
  process group, so the terminal signalled torchrun and every worker at the same moment the launcher
  forwarded a signal of its own, and the extra signal interrupted torchelastic's shutdown before it
  had reaped the workers. The launcher now starts the worker tree in a new session, forwards one
  signal to it, and escalates to ``SIGTERM`` and then ``SIGKILL`` so a worker wedged in a native call
  cannot outlive the run.
