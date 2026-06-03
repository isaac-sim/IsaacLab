.. _warp-environments:

Warp Experimental Environments
==============================

.. note::

   The warp environment infrastructure lives in ``isaaclab_experimental`` and
   ``isaaclab_tasks_experimental``. It's an experimental feature.

The experimental extensions introduce **warp-first** environment infrastructure with CUDA graph capture
support. All environment-side computation (observations, rewards, resets, actions) runs as pure Warp
kernels, eliminating Python overhead and enabling CUDA graph capture for maximum throughput.


Workflows
~~~~~~~~~

Two environment workflows are supported:

**Direct workflow** — ``DirectRLEnvWarp`` base class. You implement the step loop, observations,
rewards, and resets directly in your env class using Warp kernels.

**Manager-based workflow** — ``ManagerBasedRLEnvWarp`` base class. You define MDP terms as
standalone Warp-kernel functions and compose them via configuration.


Available Environments
~~~~~~~~~~~~~~~~~~~~~~

Direct Warp Environments
^^^^^^^^^^^^^^^^^^^^^^^^

- ``Isaac-Cartpole-Direct-Warp-v0`` — Cartpole balance
- ``Isaac-Ant-Direct-Warp-v0`` — Ant locomotion
- ``Isaac-Humanoid-Direct-Warp-v0`` — Humanoid locomotion
- ``Isaac-Repose-Cube-Allegro-Direct-Warp-v0`` — Allegro hand cube repose


Manager-Based Warp Environments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Classic**

- ``Isaac-Cartpole-Warp-v0``
- ``Isaac-Ant-Warp-v0``
- ``Isaac-Humanoid-Warp-v0``

**Locomotion (Flat)**

- ``Isaac-Velocity-Flat-Anymal-B-Warp-v0``
- ``Isaac-Velocity-Flat-Anymal-C-Warp-v0``
- ``Isaac-Velocity-Flat-Anymal-D-Warp-v0``
- ``Isaac-Velocity-Flat-Cassie-Warp-v0``
- ``Isaac-Velocity-Flat-G1-Warp-v0``
- ``Isaac-Velocity-Flat-G1-Warp-v1``
- ``Isaac-Velocity-Flat-H1-Warp-v0``
- ``Isaac-Velocity-Flat-Unitree-A1-Warp-v0``
- ``Isaac-Velocity-Flat-Unitree-Go1-Warp-v0``
- ``Isaac-Velocity-Flat-Unitree-Go2-Warp-v0``

**Manipulation**

- ``Isaac-Reach-Franka-Warp-v0``
- ``Isaac-Reach-UR10-Warp-v0``


Quick Start
~~~~~~~~~~~

.. code-block:: bash

    # Direct workflow
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Cartpole-Direct-Warp-v0 --num_envs 4096 --headless

    # Manager-based workflow
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Velocity-Flat-Anymal-C-Warp-v0 --num_envs 4096 --headless

All RL libraries with warp-compatible wrappers are supported: RSL-RL, RL Games, SKRL, and
Stable-Baselines3.


Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~

Step time comparison between the stable (torch/manager) and warp (CUDA graph captured) variants,
both running on the Newton physics backend. Measured over 300 iterations with 4096 environments.

.. note::

   The warp migration is an ongoing effort. Several components (e.g. scene write, actuator models)
   have not yet been migrated to Warp kernels and still run through torch. Further performance
   improvements are expected as these components are migrated.

.. list-table::
   :header-rows: 1
   :widths: 30 12 15 15 12

   * - Env
     - Type
     - Stable Step (us)
     - Warp Step (us)
     - Change
   * - Isaac-Cartpole-Direct
     - Direct
     - 5,274
     - 4,331
     - -17.88%
   * - Ant-Direct
     - Direct
     - 6,368
     - 3,128
     - -50.88%
   * - Humanoid-Direct
     - Direct
     - 13,937
     - 10,783
     - -22.63%
   * - Allegro-Direct
     - Direct
     - 82,950
     - 74,570
     - -10.10%
   * - Cartpole
     - Manager
     - 7,971
     - 3,642
     - -54.31%
   * - Ant
     - Manager
     - 9,781
     - 4,672
     - -52.23%
   * - Humanoid
     - Manager
     - 17,653
     - 12,505
     - -29.16%
   * - Reach-Franka
     - Manager
     - 11,458
     - 7,813
     - -31.83%
   * - Anymal-B
     - Manager
     - 29,188
     - 21,781
     - -25.38%
   * - Anymal-C
     - Manager
     - 30,938
     - 22,228
     - -28.15%
   * - Anymal-D
     - Manager
     - 32,294
     - 23,977
     - -25.75%
   * - Cassie
     - Manager
     - 17,320
     - 10,706
     - -38.19%
   * - G1
     - Manager
     - 34,487
     - 27,300
     - -20.84%
   * - H1
     - Manager
     - 22,202
     - 15,864
     - -28.55%
   * - A1
     - Manager
     - 15,257
     - 9,907
     - -35.07%
   * - Go1
     - Manager
     - 16,515
     - 11,869
     - -28.13%
   * - Go2
     - Manager
     - 15,221
     - 9,966
     - -34.52%


Which Workflows Benefit Most
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The savings come from eliminating Python / torch overhead in the env's step loop, so envs
gain in proportion to how much of their step time was previously dominated by per-kernel CPU
overhead. Reading the table above:

- **Manager-based classic RL** (Cartpole, Ant) — biggest gains (-52% to -54%). Many small
  reward / observation terms with low compute per term, so per-launch CPU overhead dominated
  the stable baseline.
- **Manager-based locomotion** (Anymal, G1, H1, Cassie, Unitree) — consistent -25% to -38%
  range. The MDP has more terms but the underlying physics step is heavier, so the relative
  Python savings shrink.
- **Direct workflow** — gains scale with how much the env's step body was Python (Ant -51%,
  Cartpole -18%, Allegro hand -10%). Direct envs that already wrote most of their work as
  GPU kernels see modest gains; ones with substantial Python state machinery see large ones.
- **Compute-heavy / scene-write-heavy envs** (Allegro hand, large humanoids) — see smaller
  relative gains because the warp-side savings are amortised over a heavier step. Components
  that still go through torch (scene write, actuator models) currently bound the floor; this
  is expected to improve as remaining components migrate to warp.

If your env's step time is dominated by physics or scene I/O, expect modest gains. If it has
many small MDP terms or a lot of Python in the step loop, expect large ones. Use the
benchmarking workflow below to measure on your task before committing to a migration.


Limitations
~~~~~~~~~~~

The warp env path is experimental and has the following known constraints. These are
specific to warp envs; for Newton physics limitations see :doc:`supported-features`.

**Physics backend**

- **Newton only.** PhysX is not supported under the warp env path. Asset and sensor
  ``class_type`` fields resolve to ``isaaclab_physx.*`` classes that depend on
  ``omni.physics.tensors`` (a Kit module the warp runtime does not initialise), and several
  warp APIs (env-mask reset, CUDA graph capture) require the Newton articulation. Configure
  the cfg with a Newton physics block (or ``physics=newton_mjwarp``).

**MDP coverage**

- Only the terms listed under :ref:`Available Warp MDP Terms <warp-env-migration>` are
  implemented. Stable envs that depend on un-migrated terms cannot be run on the warp path
  until those terms are ported.
- Some scene-side operations (asset write, actuator models, certain sensor types) still go
  through torch. They participate in the step but are not yet captured into the graph; they
  set the lower bound on observed step time.
- Sensors that depend on the Kit RTX renderer (camera-based observations) cannot be combined
  with the warp env path — they need Kit, which the warp runtime does not initialise.

**API differences vs stable**

- Reset events use a boolean ``env_mask`` (``wp.array(dtype=wp.bool)``) instead of an
  ``env_ids`` list. This is required for capture safety: variable-length indexing changes
  graph topology and breaks replay.
- All buffers must be pre-allocated in ``__init__``. There is no dynamic allocation inside
  the captured step loop, so observation / reward / termination output dimensions must be
  known at env init.
- Term functions write into a pre-allocated ``out`` buffer rather than returning a tensor.
  See :doc:`warp-env-migration` for the kernel + launch pattern.
- Code inside the captured step loop must follow capture-safety rules (no
  ``wp.to_torch``, no torch arithmetic, no lazy-evaluated properties, no Python branching
  on GPU data). See the *Capture Safety* section in :doc:`warp-env-migration` for the
  full set of rules.


Benchmarking Your Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The performance table above was produced with ``scripts/benchmarks/benchmark_rsl_rl.py``,
which runs a fixed iteration count and reports step-time statistics. Use the same script
to estimate the gain for your own task before committing to a migration.

**Single-task A/B**

.. code-block:: bash

    # Stable variant
    ./isaaclab.sh -p scripts/benchmarks/benchmark_rsl_rl.py \
        --task <Task-Name>-v0 \
        --num_envs 4096 \
        --max_iterations 500 \
        --headless \
        --benchmark_backend summary \
        --output_path benchmarks/stable

    # Warp variant — same task with -Warp- suffix
    ./isaaclab.sh -p scripts/benchmarks/benchmark_rsl_rl.py \
        --task <Task-Name>-Warp-v0 \
        --num_envs 4096 \
        --max_iterations 500 \
        --headless \
        --benchmark_backend summary \
        --output_path benchmarks/warp

The ``summary`` backend prints step time (mean / p50 / p99) and total throughput. Compare
"step time" between the two runs to estimate the gain per env step.

**Sweep across all available tasks**

``scripts/benchmarks/run_training_benchmarks.sh`` runs the full set of stable tasks listed
in the script (cartpole, ant, humanoid, locomotion, manipulation). Pair it with a
warp-tasks variant (substitute the ``-Warp-`` suffixed task ids) and diff the two outputs.

**What to look at in the output**

- *Step time (mean / p99)*: the headline number — what each env step costs.
- *Iteration time*: includes policy update; useful for end-to-end training throughput.
- *Capture overhead*: for warp runs, the first few iterations include CUDA graph capture
  cost; exclude those when comparing steady-state numbers.

**Estimating before you migrate**

If you can't run the warp variant yet (e.g. the task isn't ported), measure the stable
step time and look at where it's spent:

- ``num_envs * step_time`` dominated by physics → expect modest warp gains.
- ``step_time`` dominated by ``manager.compute_*`` calls → expect large gains, since those
  are exactly what the warp managers replace with captured kernel launches.

Use ``--num_frames`` on ``benchmark_non_rl.py`` for a no-policy step-time microbenchmark
when you want to isolate env overhead from policy compute.


Migrating Existing Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For step-by-step instructions on porting an existing stable env (or writing a new warp
env from scratch) — covering project layout, the kernel + launch pattern shared by
observations / rewards / events / terminations / actions, capture-safety rules, and
parity testing — see :doc:`warp-env-migration`.
