Benchmarking
============

Isaac Lab ships three standalone benchmark scripts that emit a common
``v1.0`` JSON schema for training-performance and startup-performance data.
The schema is defined in :mod:`isaaclab.benchmark.schema`, and the scripts
are fully usable standalone — any tool that can read JSON can consume the
output.

.. contents::
   :local:
   :depth: 2


Scripts
-------

``benchmark_startup.py``
~~~~~~~~~~~~~~~~~~~~~~~~

Profiles five IsaacLab startup phases with ``cProfile``: ``app_launch``,
``python_imports``, ``task_config``, ``env_creation``, and ``first_step``. For
each phase it records wall-clock time and the top N self-time functions.

.. code-block:: bash

   ./isaaclab.sh -p scripts/benchmarks/benchmark_startup.py \
       --task Isaac-Ant-Direct-v0 --num_envs 4096 --headless \
       --schema_v1_output /tmp/startup.json

``benchmark_rsl_rl.py``
~~~~~~~~~~~~~~~~~~~~~~~

Trains a task with the RSL-RL PPO agent and records runtime / resource /
learning metrics, including exponentially-smoothed reward and episode-length
curves.

.. code-block:: bash

   ./isaaclab.sh -p scripts/benchmarks/benchmark_rsl_rl.py \
       --task Isaac-Ant-Direct-v0 --num_envs 4096 \
       --max_iterations 500 --headless \
       --schema_v1_output /tmp/training.json

``benchmark_skrl.py``
~~~~~~~~~~~~~~~~~~~~~

The SKRL-framework counterpart to ``benchmark_rsl_rl.py``. Emits the same
schema with ``framework: "skrl"``.

.. code-block:: bash

   ./isaaclab.sh -p scripts/benchmarks/benchmark_skrl.py \
       --task Isaac-Ant-Direct-v0 --num_envs 4096 \
       --max_iterations 500 --headless \
       --schema_v1_output /tmp/training_skrl.json


v1.0 schema summary
-------------------

Each script writes a single self-contained JSON file. The shape is defined by
dataclasses in :mod:`isaaclab.benchmark.schema` — refer to the module for
per-field units and descriptions.

:class:`~isaaclab.benchmark.schema.TrainingBundle` (training scripts)
top-level keys:

* ``run`` — run identity (``run_id``, ``framework``, ``backend``, ``task``,
  ``seed``, ``num_envs``, ``max_iterations``, timestamps, ``status``).
* ``versions`` — software versions at run time (Isaac Lab, Isaac Sim, Kit,
  Newton, Warp, Torch, RSL-RL / SKRL, git metadata).
* ``hardware`` — host snapshot (hostname, GPU devices, CPU, RAM).
* ``runtime`` — aggregated timings (``iterations_completed``,
  ``iteration_time_s``, ``env_steps_per_s``, ``iterations_per_s``,
  ``startup_phase_times_s``).
* ``resources`` — aggregated GPU/CPU/RAM utilisation (mean/std/peak).
* ``learning`` — final-value and EMA-smoothed reward / episode-length curves,
  with full per-iteration series unless ``--no_series`` is passed.

:class:`~isaaclab.benchmark.schema.StartupBundle` (``benchmark_startup.py``)
replaces ``runtime`` / ``resources`` / ``learning`` with:

* ``phases`` — mapping from phase name to ``{total_time_s, top_functions}``.
* ``config`` — CLI configuration (``top_n``, ``whitelist``).


Common CLI flags
----------------

``--schema_v1_output <path>``
    Write the v1.0 JSON bundle to this path. If omitted, the script falls
    back to the legacy per-backend output format.

``--backend {physx, newton}``
    Physics backend tag recorded in the bundle. Defaults to ``physx`` if
    omitted.

``--run_id <string>``
    Explicit run-identity string. If omitted, a synthetic run_id of the
    form ``<framework>_<backend>_<task>_<YYYYMMDD-HHMMSS>_seed<seed>`` is
    generated.

``--ema_alpha <float>`` (training scripts)
    EMA smoothing factor for reward / episode-length curves (default
    ``0.05``, roughly a 20-sample window).

``--no_series`` (training scripts)
    Omit per-iteration series from the bundle, leaving only the
    ``final_raw`` + ``final_ema`` scalars.
