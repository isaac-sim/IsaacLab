:orphan:

.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _mpm-tutorials:

Exploring MPM Materials and Coupling
====================================

The material, rigid-limit, and surface studies in ``scripts/tutorials/08_mpm/``
are source-checkout tutorials. The G1 coupling comparison is a packaged
``isaaclab example`` program. Run these commands from the repository root;
each accepts ``--help`` for its options.

.. code-block:: bash

   # Compare material parameters with a fixed scene and seeded particles.
   uv run --extra isaacsim python scripts/tutorials/08_mpm/material_parameters.py \
     --preset young_modulus --visualizer kit

   # Compare nearly rigid MPM particles with MJWarp rigid bodies.
   uv run --extra isaacsim python scripts/tutorials/08_mpm/rigid_body_equivalence.py \
     --visualizer kit

   # Compare one-way and two-way coupling using the published G1 policy.
   uv run --extra isaacsim isaaclab example mpm-g1-coupling \
     --coupling two_way --visualizer kit

   # Change only the reconstruction settings of a falling water blob.
   uv run python scripts/tutorials/08_mpm/surface_reconstruction.py \
     --surface_preset balanced --visualizer newton_gl

The material tutorial has presets for stiffness, compressibility, friction,
yielding, hardening, dilatancy, viscosity, and particle jitter. Use
``--variant_index`` to run a single specimen. The G1 example downloads a
policy checkpoint unless you provide ``--checkpoint``; it requires a CUDA
device. It walks across sand, snow, and clay strips; ``--coupling one_way``
lets the robot displace particles without receiving their reaction forces.
From an installed wheel, run it with
``uvx --from 'isaaclab[isaacsim,rsl-rl]' isaaclab example mpm-g1-coupling``.
For Newton RTX from a source checkout, add ``--extra ovrtx`` to ``uv run``.

Read ``scripts/tutorials/08_mpm/README.md`` for the controlled settings and
interpretation limits. For scene authoring
and solver configuration, see :ref:`newton-using-mpm`.
