Fixed
^^^^^

* Fixed multi-GPU training saving ``params/env.yaml`` and ``params/agent.yaml`` with the seed of
  whichever rank wrote last, and leaving extra settings-only folders when ranks started in different
  seconds. All ranks of a ``train_multigpu`` launch now shared one run folder, and each rank wrote its
  own outputs (``params/``, videos, and sensor captures) to ``rank_<rank>/`` inside it instead of
  overwriting each other's files; ``rank_0/params/`` held the launch settings. Single-GPU runs kept their
  layout. For multi-node jobs on shared storage, export the same ``ISAACLAB_RUN_TIMESTAMP`` on every node.
* Fixed skrl multi-GPU training saving the final checkpoint from every rank to the same file.
* Fixed multi-GPU training ending with a ``destroy_process_group() was not called`` warning after a
  successful run.
