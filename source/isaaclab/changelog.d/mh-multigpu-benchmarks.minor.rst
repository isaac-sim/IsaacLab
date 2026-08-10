Added
^^^^^

* Added multi-GPU benchmark workflows. ``isaaclab benchmark startup-multigpu``,
  ``runtime-multigpu``, and ``training-multigpu`` launch one rank per GPU with ``torchrun`` and
  accept the same ``--num_gpus``, ``--nnodes``, and rendezvous options as ``train_multigpu``.
  ``--num_envs`` is per rank, and only global rank 0 writes a bundle.
* Added :mod:`isaaclab.benchmark.distributed`, whose :class:`~isaaclab.benchmark.distributed.DistributedContext`
  carries the rank layout, scales one rank's workload to the global workload, and records the
  measurement scope of a distributed bundle in ``extra``.
* Added :mod:`isaaclab.cli.multigpu`, the shared ``torchrun`` launcher used by both the training
  and the benchmark multi-GPU commands.
* Added ``Resources.devices`` to the benchmark schema (version 1.4), reporting utilisation and
  memory for every CUDA device visible to the run rather than only the device it used.

Changed
^^^^^^^

* Changed the RSL-RL, RL-Games, and skrl training benchmark adapters to accept ``--distributed``
  instead of rejecting it. A distributed run reports global throughput and workload counts with
  rank-0 timing, learning curves, and resource usage; video recording, environment sensor capture,
  and success-based early stopping are rejected because they are not rank-safe.

Fixed
^^^^^

* Fixed :class:`~isaaclab.benchmark.BaseIsaacLabBenchmark` to tolerate concurrent processes creating
  the same output directory, which previously failed when several ranks shared ``--output_path``.
