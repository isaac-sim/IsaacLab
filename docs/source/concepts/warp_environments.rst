.. _warp-environments:

Warp Environments
=================

.. note::

   The warp environment infrastructure lives in ``isaaclab_experimental`` and
   ``isaaclab_tasks_experimental``. It's an experimental feature.

Experimental Warp environments compute MDP terms (observations, rewards, terminations, events, commands,
and actions) in Warp kernels and record the capturable terms of each manager stage into CUDA graphs, which
reduces Python overhead during execution. Scene writes, actuator models, the stable curriculum and recorder
managers, and terms marked non-capturable still run eagerly.

Here, **stable** means the default Torch environments in ``isaaclab`` and
``isaaclab_tasks``. Pass ``--frontend warp`` to use their experimental Warp counterparts.


Quick Start
~~~~~~~~~~~

.. code-block:: bash

    # Direct workflow: stable task, warp env implementation
    uv run isaaclab train --rl_library rsl_rl \
        --task Isaac-Cartpole-Direct --frontend warp presets=newton_mjwarp --num_envs 4096
    uv run isaaclab play --rl_library rsl_rl \
        --task Isaac-Cartpole-Direct --frontend warp presets=newton_mjwarp \
        --num_envs 32 --checkpoint latest --visualizer newton_gl

    # Manager-based workflow: stable task on the warp runtime
    uv run isaaclab train --rl_library rsl_rl \
        --task Isaac-Velocity-Flat-AnymalD --frontend warp presets=newton_mjwarp --num_envs 4096
    uv run isaaclab play --rl_library rsl_rl \
        --task Isaac-Velocity-Flat-AnymalD --frontend warp presets=newton_mjwarp \
        --num_envs 32 --checkpoint latest --visualizer newton_gl

All RL libraries with warp-compatible wrappers are supported: RSL-RL, RL Games, SKRL, and
Stable-Baselines3.

.. note::

   ``--video`` is rejected on the warp path, for both ``train`` and ``play``: video
   recording requires the standard torch frontend. To record a rollout, replay the same
   checkpoint with ``--frontend torch``; see :ref:`how_to_record_video`.


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

Direct tasks share the stable task configuration. ``--frontend warp`` resolves
the stable ``<task>_direct_env:<Name>Env`` entry point to the mirrored
``<task>_warp_env:<Name>WarpEnv`` implementation and swaps only the environment
class. Registrations can use ``warp_entry_point`` as an optional override when
the implementation cannot follow this convention. Stable tasks with a Warp
implementation:

- ``Isaac-Cartpole-Direct`` — Cartpole balance
- ``Isaac-Ant-Direct`` — Ant locomotion
- ``Isaac-Humanoid-Direct`` — Humanoid locomotion
- ``Isaac-Reorient-Cube-Allegro-Direct`` — Allegro hand cube reorient


Manager-Based Warp Execution (``--frontend warp``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Manager-based tasks do not need a parallel warp registration: the shared RL
CLI exposes ``--frontend {torch,warp}``, and ``--frontend warp`` adapts the
*stable* task configuration onto the warp runtime at build time (swapping each
MDP term for its warp twin). Select the Newton solver explicitly with
``presets=newton_mjwarp``. Stable tasks with full twin coverage:

- ``Isaac-Cartpole``
- ``Isaac-Ant``
- ``Isaac-Humanoid``
- ``Isaac-Reach-Franka``
- ``Isaac-Reach-UR10``
- ``Isaac-Velocity-Flat-AnymalD``
- ``Isaac-Velocity-Flat-Cassie``
- ``Isaac-Velocity-Flat-G1``
- ``Isaac-Velocity-Flat-H1``
- ``Isaac-Velocity-Flat-UnitreeGo2``
- ``Isaac-Velocity-Rough-AnymalD``
- ``Isaac-Velocity-Rough-Cassie``
- ``Isaac-Velocity-Rough-G1``
- ``Isaac-Velocity-Rough-H1``
- ``Isaac-Velocity-Rough-UnitreeGo2``

The following contributed tasks also have full twin coverage:

- ``IsaacContrib-Velocity-Flat-AnymalB``
- ``IsaacContrib-Velocity-Flat-AnymalC``
- ``IsaacContrib-Velocity-Flat-UnitreeA1``
- ``IsaacContrib-Velocity-Flat-UnitreeGo1``
- ``IsaacContrib-Velocity-Rough-AnymalB``
- ``IsaacContrib-Velocity-Rough-AnymalC``
- ``IsaacContrib-Velocity-Rough-UnitreeA1``
- ``IsaacContrib-Velocity-Rough-UnitreeGo1``

A missing twin is a hard error listing the affected terms, so a partially
covered task fails at build time rather than silently changing behavior.
The Rough tasks keep the stable terrain curriculum. Their ``height_scan``
observation, and ``base_height_l2`` when it reads the height scanner, read a ray
caster whose refresh is decided on the host, so these terms run eagerly.


Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~

These historical measurements compare Torch manager-based and CUDA-graph-captured
Warp environments on Newton, over 300 iterations with 4096 environments. They do
not represent the latest revision or every supported task.

For a current comparison, use the same hardware and Isaac Lab, Newton, and Warp
revisions for both frontends. Shared-library improvements can benefit both.

.. note::

   The warp migration is an ongoing effort. Several components (e.g. scene write, actuator models)
   have not yet been migrated to Warp kernels and still run through torch. Further performance
   improvements are expected as these components are migrated.

.. list-table::
   :header-rows: 1
   :widths: 42 10 14 14 12

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
   * - Isaac-Ant-Direct
     - Direct
     - 6,368
     - 3,128
     - -50.88%
   * - Isaac-Humanoid-Direct
     - Direct
     - 13,937
     - 10,783
     - -22.63%
   * - Isaac-Reorient-Cube-Allegro-Direct
     - Direct
     - 82,950
     - 74,570
     - -10.10%
   * - Isaac-Cartpole
     - Manager
     - 7,971
     - 3,642
     - -54.31%
   * - Isaac-Ant
     - Manager
     - 9,781
     - 4,672
     - -52.23%
   * - Isaac-Humanoid
     - Manager
     - 17,653
     - 12,505
     - -29.16%
   * - Isaac-Reach-Franka
     - Manager
     - 11,458
     - 7,813
     - -31.83%
   * - Isaac-Velocity-Flat-AnymalD
     - Manager
     - 32,294
     - 23,977
     - -25.75%
   * - Isaac-Velocity-Flat-Cassie
     - Manager
     - 17,320
     - 10,706
     - -38.19%
   * - Isaac-Velocity-Flat-G1
     - Manager
     - 34,487
     - 27,300
     - -20.84%
   * - Isaac-Velocity-Flat-H1
     - Manager
     - 22,202
     - 15,864
     - -28.55%
   * - Isaac-Velocity-Flat-UnitreeGo2
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
- **Manager-based locomotion** (AnymalD, G1, H1, Cassie, UnitreeGo2) — consistent -21% to -38%
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
specific to warp envs; for Newton physics features and related guides see
:ref:`physics-backends-newton`.

**Physics backend**

- **Newton only.** PhysX is not supported under the warp env path. Asset and sensor
  ``class_type`` fields resolve to ``isaaclab_physx.*`` classes that depend on
  ``omni.physics.tensors`` (a Kit module the warp runtime does not initialise), and several
  warp APIs (env-mask reset, CUDA graph capture) require the Newton articulation. Configure
  the cfg with a Newton physics block (or the typed selector ``physics=newton_mjwarp``,
  which fails loudly if the task has no Newton physics preset).

**MDP coverage**

- Only the terms listed under :ref:`Available Warp MDP Terms <warp-env-mdp-terms>` are
  implemented. Stable envs that depend on un-migrated terms cannot be run on the warp path
  until those terms are ported.
- Some scene-side operations (asset write, actuator models, certain sensor types) still go
  through torch. They participate in the step but are not yet captured into the graph; they
  set the lower bound on observed step time.
- A term marked ``@WarpCapturable(False)``, or rejected by its parameter predicate, runs eagerly
  between the recorded graphs of its stage. Observation, reward, termination, command and action
  stages run their eager terms before the recorded ones; event stages keep the configured order, so
  an eager event term splits its stage into several graphs.
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
  See :ref:`warp-env-migration` for the kernel + launch pattern.
- Code inside the captured step loop must follow capture-safety rules (no
  ``wp.to_torch``, no torch arithmetic, no lazy-evaluated properties, no Python branching
  on GPU data). See :ref:`warp-env-capture-safety` for the
  full set of rules.


Benchmarking Your Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The performance table above was produced with ``isaaclab benchmark training``,
which runs a fixed iteration count and reports step-time statistics. Use the same command
to estimate the gain for your own task before committing to a migration.

**Single-task A/B**

Run the same stable task id twice, differing only in ``--frontend``. Pass
``presets=newton_mjwarp`` to *both* runs so the physics backend is identical and the
measured difference isolates the frontend (manager pipeline + CUDA graph capture):

.. code-block:: bash

    # Torch frontend (stable managers)
    uv run isaaclab benchmark training \
        --rl_library rsl_rl \
        --task <Task-Name> \
        --num_envs 4096 \
        --max_iterations 500 \
        --benchmark_formatter summary \
        --output_path benchmarks/torch \
        presets=newton_mjwarp

    # Warp frontend (warp managers, CUDA-graph captured) — same task id
    uv run isaaclab benchmark training \
        --rl_library rsl_rl \
        --task <Task-Name> \
        --frontend warp \
        --num_envs 4096 \
        --max_iterations 500 \
        --benchmark_formatter summary \
        --output_path benchmarks/warp \
        presets=newton_mjwarp

The ``summary`` formatter prints step time (min / mean / max) and total throughput. Compare
"step time" between the two runs to estimate the gain per env step.

**Sweep across all available tasks**

Run ``isaaclab benchmark training`` for each task in the stable set (cartpole, ant, humanoid,
locomotion, manipulation) and again with ``--frontend warp``, then diff the two output
directories.

**What to look at in the output**

- *Step time (min / mean / max)*: the headline number — what each env step costs.
- *Iteration time*: includes policy update; useful for end-to-end training throughput.
- *Capture overhead*: for warp runs, the first few iterations include CUDA graph capture
  cost; exclude those when comparing steady-state numbers.

**Estimating before you migrate**

If you can't run the warp variant yet (e.g. the task isn't ported), measure the stable
step time and look at where it's spent:

- ``num_envs * step_time`` dominated by physics → expect modest warp gains.
- ``step_time`` dominated by ``manager.compute_*`` calls → expect large gains, since those
  are exactly what the warp managers replace with captured kernel launches.

Use ``--num_steps`` on ``runtime.py`` for a no-policy step-time microbenchmark
when you want to isolate env overhead from policy compute.


.. _warp-env-migration:

Migrating Existing Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following sections cover project layout, Warp kernels, CUDA graph capture,
and parity testing for new or migrated environments.


Design Rationale
^^^^^^^^^^^^^^^^

The warp environment path is built around `CUDA graph capture
<https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html>`_.
A CUDA graph records a sequence of GPU operations (kernel launches, memory copies) during a
capture phase, then replays the entire sequence with a single launch. This eliminates per-kernel
CPU overhead — the parameter validation, kernel selection, and buffer setup that normally costs
20–200 μs per operation is performed once during graph instantiation and reused on every replay
(~10 μs total). All CPU-side code (Python logic, torch dispatching) executed during capture is
completely bypassed during replay. See the `Warp concurrency documentation
<https://nvidia.github.io/warp/stable/deep_dive/concurrency.html>`_ for Warp's graph capture API
(``wp.ScopedCapture``).

All design decisions in the warp infrastructure follow from this constraint: every operation in the
step loop must be a GPU kernel launch with stable memory pointers so that the captured graph can
be replayed without modification.

Key consequences:

- All buffers are **pre-allocated** — no dynamic allocation inside the step loop
- Data flows through **persistent ``wp.array`` pointers** — never replaced, only overwritten
- MDP terms are **pure ``@wp.kernel`` functions** — no Python branching on GPU data
- Reset uses **boolean masks** (``env_mask``) instead of index lists (``env_ids``) to avoid
  variable-length indexing that changes graph topology


Project Structure
^^^^^^^^^^^^^^^^^

Warp-specific implementations that diverge from the torch-based managers and env classes live in the ``_experimental`` packages:

- ``isaaclab_experimental`` — warp managers, base env classes, warp MDP terms
- ``isaaclab_tasks_experimental`` — task-specific warp MDP terms and direct warp envs

Task configurations are not duplicated: ``--frontend warp`` builds the environment from the stable task
configuration with Newton physics selected. The experimental packages mirror the stable module tree, and each
stable term is replaced by the same-named warp term of the mirrored ``mdp`` package (its *twin*). A term
without a twin is a build-time error.


Writing Warp MDP Terms
^^^^^^^^^^^^^^^^^^^^^^

Imports
"""""""

Warp terms import the scene entity configuration from the stable managers and the warp managers' helpers from the
experimental packages:

.. code-block:: python

   from isaaclab.managers import SceneEntityCfg
   from isaaclab_experimental.managers import ManagerTermBase
   from isaaclab.utils.warp.capture import WarpCapturable

Term configurations keep the stable config classes (for example
:class:`~isaaclab.managers.RewardTermCfg` and the command configurations in :mod:`isaaclab.envs.mdp`);
the frontend only replaces their ``func`` or ``class_type``.


Common Pattern
""""""""""""""

All warp MDP terms (observations, rewards, terminations, events, actions) follow the same
**kernel + launch** pattern. Torch terms use torch tensors and return results; warp terms
write into pre-allocated ``wp.array`` output buffers via ``@wp.kernel`` functions:

.. code-block:: python

   # Torch — returns a tensor
   def lin_vel_z_l2(env, asset_cfg) -> torch.Tensor:
       return torch.square(asset.data.root_lin_vel_b[:, 2])

   # Warp — writes into pre-allocated output
   @wp.kernel
   def _lin_vel_z_l2_kernel(vel: wp.array(...), out: wp.array(dtype=wp.float32)):
       i = wp.tid()
       out[i] = vel[i][2] * vel[i][2]

   def lin_vel_z_l2(env, out, asset_cfg) -> None:
       wp.launch(_lin_vel_z_l2_kernel, dim=env.num_envs, inputs=[..., out])

The output buffer shapes differ by term type:

- **Observations**: ``(num_envs, D)`` where D is the observation dimension
- **Rewards**: ``(num_envs,)``
- **Terminations**: ``(num_envs,)`` with dtype ``bool``
- **Events**: ``(num_envs,)`` mask — events don't produce output, they modify sim state


Observation Terms
"""""""""""""""""

Since warp terms write into pre-allocated buffers, the observation manager must know each
term's output dimension at initialization to allocate the correct ``(num_envs, D)`` output
array. This is resolved via a fallback chain (see
``ObservationManager._infer_term_dim_scalar`` in
``isaaclab_experimental/managers/observation_manager.py``):

.. warning::

   The IO descriptor decorators are deprecated in Isaac Lab 3.0 and will be
   removed in Isaac Lab 3.2. Their output-dimension metadata remains available
   temporarily for Warp-first environments while a replacement runtime
   configuration is developed.

1. **Explicit ``out_dim`` in decorator** (preferred):

   .. code-block:: python

      @generic_io_descriptor_warp(out_dim=3, observation_type="RootState")
      def base_lin_vel(env, out, asset_cfg) -> None: ...

   ``out_dim`` can be an integer, or a string that resolves at initialization:

   - ``"joint"`` — number of selected joints from ``asset_cfg``
   - ``"body:N"`` — N components per selected body from ``asset_cfg``
   - ``"command"`` — dimension from command manager
   - ``"action"`` — dimension from action manager

2. **``axes`` metadata**: Dimension equals the number of axes listed:

   .. code-block:: python

      @generic_io_descriptor_warp(axes=["X", "Y", "Z"], observation_type="RootState")
      def projected_gravity(env, out, asset_cfg) -> None: ...
      # → dimension = 3

3. **Legacy params**: ``term_dim``, ``out_dim``, or ``obs_dim`` keys in ``term_cfg.params``.

4. **Ray-caster fallback**: Number of rays of the sensor in ``sensor_cfg``, for terms that
   write one value per ray (e.g. ``height_scan``).

5. **Asset config fallback**: Count of ``asset_cfg.joint_ids`` (or ``joint_ids_wp``) for
   joint-level terms.


Event Terms
"""""""""""

Events use ``env_mask`` (boolean ``wp.array``) instead of ``env_ids``, and each kernel
checks the mask to skip non-selected environments:

.. code-block:: python

   def reset_joints_by_offset(env, env_mask, ...):
       wp.launch(_kernel, dim=env.num_envs, inputs=[env_mask, ...])

   @wp.kernel
   def _kernel(env_mask: wp.array(dtype=wp.bool), ...):
       i = wp.tid()
       if not env_mask[i]:
           return
       # ... modify state for selected envs only

- RNG uses the per-env :attr:`WarpRng.state <isaaclab.utils.seed.WarpRng.state>` (``wp.uint32``) instead of
  ``torch.rand``
- **Startup/prestartup** events use the torch convention ``(env, env_ids, **params)``
- **Reset/interval** events use the warp convention ``(env, env_mask, **params)``


Action Terms
""""""""""""

Actions follow a **two-stage execution**: ``process_actions`` (called once per env step) scales
and clips raw actions, and ``apply_actions`` (called once per sim step) writes targets to the
asset. Both stages use warp kernels with pre-allocated ``_raw_actions`` and ``_processed_actions``
buffers.


.. _warp-env-capture-safety:

Command Terms
"""""""""""""

Command terms subclass :class:`isaaclab_experimental.managers.CommandTerm`. They keep their command and
metrics in persistent ``wp.array`` buffers and resample the environments selected by a mask with
:attr:`WarpRng.state <isaaclab.utils.seed.WarpRng.state>`; the command manager records ``compute`` and
``reset`` like the other managers.
Warp terms read a command with ``env.command_manager.get_command_wp(name)`` on every call instead of
caching it, so a term never holds another environment's buffer.


Capture Safety
""""""""""""""

Each manager method decorated with :func:`~isaaclab.utils.warp.capture.captured` is a stage (for example
``RewardManager.compute``, or the ``_reset`` that ``CommandManager.reset`` calls with the reset mask): its
first call records it into CUDA graphs and later calls replay them. That first call is the recording
itself; there is no eager warm-up. A stage records again when it receives a different array or scalar
argument, and every graph is dropped when the physics backend rebinds its buffers or stops. Only device
work is replayed: Python in a recorded term runs once, while the stage records.

Keeping a term safe to record is the author's contract; a synchronization check catches only
part of it:

- **Allocate in** ``__init__`` — class terms allocate every buffer before their first call and
  function terms allocate nothing. A buffer allocated while recording belongs to the graph.
- **Write into persistent buffers** — outputs, metrics and state are ``wp.array`` buffers that are
  overwritten in place, never replaced. Values returned for logging, for example from ``reset``,
  are persistent views over such buffers.
- **Select environments with** ``env_mask`` — kernels launch over ``num_envs`` and test the mask.
  Converting a mask to indices (``nonzero``) or reading device values on the host (``.item()``,
  ``.cpu()``, ``.numpy()``) synchronizes the device and cannot be recorded. Only the environment
  base classes do it, once per step, to decide whether any environment resets.
- **Read asset data when called** — pass sim-bound (Tier 1) buffers such as ``root_link_pose_w``
  and ``root_com_vel_w`` to the kernel and compute derived quantities inline. Lazily computed
  properties refresh through a host-side timestamp that does not replay, and a view cached across
  a rebind points at freed memory.
- **Change settings through the managers** — a scalar read in Python, such as a configuration value
  or ``env.common_step_counter``, is frozen into the graph when the stage records. The managers'
  ``set_term_cfg`` records their stages again when it changes what they read from a term's configuration
  (observation terms are named ``"<group>/<term>"``), and the ``modify_reward_weight`` and
  ``modify_term_cfg`` curricula go through it, so an observation noise curriculum applies to recorded
  observations. Setting an unchanged configuration keeps the recorded stages. Reward weights are read on
  the device, so a weight change applies without recording again, unless the weight turns zero or nonzero,
  which takes the term out of the stage or puts it back. A value that changes every step belongs in a
  device buffer that the kernel reads. Writing a term configuration directly, bypassing ``set_term_cfg``,
  is not supported while stages are recorded.
- **Mark host-dependent terms** — a term that relies on host-side work, such as a sensor whose
  refresh is decided on the host, is decorated with ``@WarpCapturable(False, reason=...)``. Its stage
  calls it through :func:`~isaaclab.utils.warp.capture.eager`, between recorded graphs, on every call,
  and it raises if it is ever called while a stage records. When only some configurations need the host,
  pass a predicate over the term's parameters, e.g.
  ``@WarpCapturable(lambda params: params.get("sensor_cfg") is None, reason=...)``. On a class term the
  annotation covers its call; decorate its ``reset`` method to keep the reset out of the recording. A
  command term's class annotation covers both its ``compute`` and its ``reset``. ``height_scan``,
  ``base_height_l2`` with a height scanner and the rigid-body randomization events are examples.

Setting :attr:`CapturedStage.enabled <isaaclab.utils.warp.capture.CapturedStage.enabled>` to ``False``
after the environment is built runs every stage eagerly, and ``ISAACLAB_SYNC_DEBUG=1`` makes a hidden
device-to-host synchronization inside an eager stage raise. Comparing an eager and a captured
rollout of the same seeded task is the end-to-end check.

Terms from Other Packages
"""""""""""""""""""""""""

Terms defined outside the mirrored packages, for example in an external project, have no twins.
The frontend uses them unchanged when they are warp terms: a class term that subclasses
:class:`isaaclab_experimental.managers.ManagerTermBase` (warp action and command terms included),
or a function term that declares its capture behavior with ``@WarpCapturable(True)`` or
``@WarpCapturable(False, reason=...)``. Any other term from such a package is rejected, even when a
built-in warp term has the same name.

.. code-block:: python

   import warp as wp

   from isaaclab.utils.warp.capture import WarpCapturable


   @wp.kernel
   def _upright_kernel(pose: wp.array(dtype=wp.transformf), out: wp.array(dtype=wp.float32)):
       i = wp.tid()
       up = wp.quat_rotate(wp.transform_get_rotation(pose[i]), wp.vec3f(0.0, 0.0, 1.0))
       out[i] = up[2]


   @WarpCapturable(True)
   def upright_bonus(env, out, asset_cfg) -> None:
       asset = env.scene[asset_cfg.name]
       wp.launch(_upright_kernel, dim=env.num_envs, inputs=[asset.data.root_link_pose_w.warp, out])


Parity Testing
^^^^^^^^^^^^^^

Two levels of parity testing are used to validate warp terms:

**1. Implementation parity (torch vs warp)** — verifies that the warp kernel produces the
same result as the torch implementation. This is optional for terms that have no torch
counterpart (e.g. new terms written directly in warp).

.. code-block:: python

   import isaaclab.envs.mdp.observations as torch_obs
   import isaaclab_experimental.envs.mdp.observations as warp_obs

   # Torch baseline
   expected = torch_obs.joint_pos(torch_env, asset_cfg=cfg)

   # Warp (uncaptured)
   out = wp.zeros((num_envs, num_joints), dtype=wp.float32, device=device)
   warp_obs.joint_pos(warp_env, out, asset_cfg=cfg)
   actual = wp.to_torch(out)

   torch.testing.assert_close(actual, expected)

**2. Capture parity (warp vs warp-captured)** — verifies that the term produces identical
results when replayed from a CUDA graph vs launched directly. A mismatch here indicates capture-unsafe
code (e.g. stale pointers, dynamic allocation, or lazy property access that doesn't replay).
This test should always be run, even for terms without a torch counterpart.

.. code-block:: python

   # Warp uncaptured
   out_uncaptured = wp.zeros((num_envs, num_joints), dtype=wp.float32, device=device)
   warp_obs.joint_pos(warp_env, out_uncaptured, asset_cfg=cfg)

   # Warp captured (graph replay)
   out_captured = wp.zeros((num_envs, num_joints), dtype=wp.float32, device=device)
   with wp.ScopedCapture() as cap:
       warp_obs.joint_pos(warp_env, out_captured, asset_cfg=cfg)
   wp.capture_launch(cap.graph)

   torch.testing.assert_close(wp.to_torch(out_captured), wp.to_torch(out_uncaptured))

See ``source/isaaclab_experimental/test/envs/mdp/`` for complete parity test examples.
``test_capture_safety.py`` discovers every warp MDP term: a new capturable term needs a ``CAPTURE_SPECS``
entry there, which records the term, overwrites its inputs in place, replays it, and compares the result
with the stable term on the new data.


.. _warp-env-mdp-terms:

Available Warp MDP Terms
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Category
     - Available Terms
   * - Observations (12)
     - | ``base_pos_z``
       | ``base_lin_vel``
       | ``base_ang_vel``
       | ``projected_gravity``
       | ``joint_pos``
       | ``joint_pos_rel``
       | ``joint_pos_limit_normalized``
       | ``joint_vel``
       | ``joint_vel_rel``
       | ``last_action``
       | ``generated_commands``
       | ``height_scan``
   * - Rewards (16)
     - | ``is_alive``
       | ``is_terminated``
       | ``lin_vel_z_l2``
       | ``ang_vel_xy_l2``
       | ``flat_orientation_l2``
       | ``joint_torques_l2``
       | ``joint_vel_l1``
       | ``joint_vel_l2``
       | ``joint_acc_l2``
       | ``joint_deviation_l1``
       | ``joint_pos_limits``
       | ``action_rate_l2``
       | ``action_l2``
       | ``undesired_contacts``
       | ``track_lin_vel_xy_exp``
       | ``track_ang_vel_z_exp``
   * - Events (6)
     - | ``reset_joints_by_offset``
       | ``reset_joints_by_scale``
       | ``reset_root_state_uniform``
       | ``push_by_setting_velocity``
       | ``apply_external_force_torque``
       | ``randomize_rigid_body_com``
   * - Terminations (4)
     - | ``time_out``
       | ``root_height_below_minimum``
       | ``joint_pos_out_of_manual_limit``
       | ``illegal_contact``
   * - Actions (2)
     - | ``JointPositionAction``
       | ``JointEffortAction``
   * - Commands (3)
     - | ``UniformVelocityCommand``
       | ``UniformPoseCommand``
       | ``NullCommand``

Terms not listed here remain in torch only. When using an env that requires unlisted terms,
those terms must be implemented in warp first.
