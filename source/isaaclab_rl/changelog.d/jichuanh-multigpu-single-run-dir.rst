Fixed
^^^^^

* Fixed multi-GPU training saving ``params/env.yaml`` and ``params/agent.yaml`` with the seed of
  whichever rank wrote last, and leaving extra settings-only folders when ranks started in different
  seconds. All ranks of a ``train_multigpu`` launch now shared one run folder: rank 0 wrote its settings,
  videos, and sensor captures where a single-GPU run does, and every other rank wrote its own ``params/``,
  videos, and sensor captures to ``rank_<rank>/``. For multi-node jobs on shared storage, pass the same
  ``--run_timestamp`` to ``train_multigpu`` on every node.
* Fixed skrl multi-GPU training saving the final checkpoint from every rank to the same file.
* Fixed multi-GPU training ending with a ``destroy_process_group() was not called`` warning after a
  successful run.
* Fixed resumed RSL-RL multi-GPU training loading the checkpoint onto the GPU that saved it in every
  process instead of the process's own GPU, which could make the first update fail with
  ``normal expects all elements of std >= 0.0``.
