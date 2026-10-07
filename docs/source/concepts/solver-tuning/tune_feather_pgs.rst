.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _newton-feather-pgs-solver:
.. _feather-pgs-solver-tuning:

Tune FeatherPGS
===============

FeatherPGS is an experimental Newton solver selected through a task-exposed ``feather_pgs`` preset. It
integrates articulations in reduced coordinates and solves contacts, joint limits and, optionally, joint drives
with projected Gauss-Seidel iterations. The generated API documentation for
:class:`~isaaclab_newton.physics.FeatherPGSSolverCfg` remains authoritative for current fields and defaults.

Prerequisites
-------------

Read :ref:`backends-and-presets` for preset semantics and prepare the asset with
:doc:`/source/how-to/prepare_asset_for_newton`. Confirm that the task exposes ``feather_pgs`` through its
``--help`` output; do not infer FeatherPGS support from another Newton preset. The default ``pgs_mode="matrix_free"``
requires a CUDA device; use ``pgs_mode="split"`` on CPU.

Start from an explicit baseline
-------------------------------

Run one small smoke test before training or benchmarking:

.. code-block:: bash

    uv run python scripts/environments/zero_agent.py --task Isaac-Cartpole-Direct --num_envs 4 --viz newton physics=feather_pgs

Then fix the initial state, seed, action sequence and reset distribution, and change one setting at a time.

Size the constraint capacities
------------------------------

FeatherPGS allocates fixed per-world row capacities: ``dense_max_constraints`` for rows that involve
articulated bodies and ``mf_max_constraints`` for free-body contact rows. Rows beyond a capacity are dropped.
``warn_constraint_overflow`` prints a warning the first time a world overflows; set
``raise_on_constraint_overflow=True`` while sizing a task so a dropped row fails the step instead. Size the
rigid-contact buffer with :attr:`~isaaclab_newton.physics.NewtonCollisionPipelineCfg.rigid_contacts_per_world`
or ``rigid_contact_max``; FeatherPGS allocates its contact scratch from that capacity.

Tune the solve
--------------

Increase ``pgs_iterations`` when contacts or limits do not converge, and lower ``pgs_beta`` when position
correction injects energy. ``contact_gap_gate`` drops rows of contacts that are still separated by more than
the gate. Predictive contacts are enabled with
:attr:`~isaaclab_newton.physics.FeatherPGSSolverCfg.speculative_contact_gap_max`; the collision pipeline then
predicts contacts over the full physics step.

``contact_torsion_radius`` adds spin friction to the contacts of articulated bodies. Set
``contact_torsion_device=True`` to keep CUDA graph capture; host-prepared torsion steps eagerly. Torsion errors
are raised after the step that produced them, before the simulation time advances. ``enable_sleeping`` freezes
supported, quiet islands of articulations until a contact, force or reset wakes them.

Determinism and contact matching
--------------------------------

FeatherPGS does not provide Newton's solver determinism guarantee, so
:attr:`~isaaclab_newton.physics.NewtonCfg.deterministic_mode` rejects it. Set
:attr:`~isaaclab_newton.physics.NewtonCollisionPipelineCfg.deterministic` to sort contacts, and
:attr:`~isaaclab_newton.physics.NewtonCollisionPipelineCfg.contact_matching` for frame-to-frame contact matching.
``"sticky"`` matching replays saved contact geometry, so it changes contacts even without ``pgs_warmstart``.

Limitations
-----------

FeatherPGS does not simulate tendons; imported tendon metadata is kept, but tendon targets cannot be commanded.
Passive joint springs are not applied.
