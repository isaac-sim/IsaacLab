.. _warp-env-migration:

Warp Environment Migration Guide
================================

This guide covers the key conventions and patterns used by the warp-first environment
infrastructure, useful for migrating existing torch environments or creating new ones
natively. For an overview of the warp env path itself (workflows, available envs,
performance, limitations, benchmarking), see :doc:`warp-environments`.


Design Rationale
~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~

Warp-specific implementations that diverge from the torch-based managers and env classes live in the ``_experimental`` packages:

- ``isaaclab_experimental`` — warp managers, base env classes, warp MDP terms
- ``isaaclab_tasks_experimental`` — warp task configs and task-specific MDP terms

Any new warp implementation that differs from the torch-based managers or env classes belongs in these packages.
Warp task configs reference Newton physics directly (no ``PresetCfg``) since the warp path
is Newton-only.


Writing Warp MDP Terms
~~~~~~~~~~~~~~~~~~~~~~

Imports
^^^^^^^

Warp task configs import from the experimental packages:

.. code-block:: python

   # Warp
   from isaaclab_experimental.managers import ObservationTermCfg, RewardTermCfg, SceneEntityCfg
   import isaaclab_experimental.envs.mdp as mdp

The term config classes have the same interface — only the import path changes.


Common Pattern
^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^

Since warp terms write into pre-allocated buffers, the observation manager must know each
term's output dimension at initialization to allocate the correct ``(num_envs, D)`` output
array. This is resolved via a fallback chain (see
``ObservationManager._infer_term_dim_scalar`` in
``isaaclab_experimental/managers/observation_manager.py``):

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

4. **Asset config fallback**: Count of ``asset_cfg.joint_ids`` (or ``joint_ids_wp``) for
   joint-level terms.


Event Terms
^^^^^^^^^^^

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

- RNG uses per-env ``env.rng_state_wp`` (``wp.uint32``) instead of ``torch.rand``
- **Startup/prestartup** events use the torch convention ``(env, env_ids, **params)``
- **Reset/interval** events use the warp convention ``(env, env_mask, **params)``


Action Terms
^^^^^^^^^^^^

Actions follow a **two-stage execution**: ``process_actions`` (called once per env step) scales
and clips raw actions, and ``apply_actions`` (called once per sim step) writes targets to the
asset. Both stages use warp kernels with pre-allocated ``_raw_actions`` and ``_processed_actions``
buffers.


Capture Safety
^^^^^^^^^^^^^^

When writing terms that run inside the captured step loop, keep in mind:

- **No ``wp.to_torch``** or torch arithmetic — stay in warp throughout
- **No lazy-evaluated properties** — use sim-bound (Tier 1) data directly; if a derived
  quantity is needed, compute it inline in the kernel
- **No dynamic allocation** — all buffers must be pre-allocated in ``__init__``


Parity Testing
~~~~~~~~~~~~~~

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


Available Warp MDP Terms
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Category
     - Available Terms
   * - Observations (11)
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

Terms not listed here remain in torch only. When using an env that requires unlisted terms,
those terms must be implemented in warp first.
