.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _mpm-tutorials:

Exploring MPM Materials and Coupling
====================================

The `MPM tutorial scripts <https://github.com/isaac-sim/IsaacLab/tree/develop/scripts/tutorials/08_mpm>`_
are source-checkout examples, not packaged ``isaaclab example`` programs. Run
them from the repository root. Each script accepts ``--help`` and can be
changed to study a different scene or parameter range.

.. code-block:: bash

   # Compare material parameters with a fixed scene and seeded particles.
   uv run python scripts/tutorials/08_mpm/material_parameters.py \
     --preset young_modulus --visualizer kit

   # Compare nearly rigid MPM particles with MJWarp rigid bodies.
   uv run python scripts/tutorials/08_mpm/rigid_body_equivalence.py \
     --visualizer kit

   # Compare one-way and two-way coupling using the published G1 policy.
   uv run --extra rsl-rl python scripts/tutorials/08_mpm/g1_coupling.py \
     --coupling two_way --visualizer kit

   # Change only the reconstruction settings of a falling water blob.
   uv run python scripts/tutorials/08_mpm/surface_reconstruction.py \
     --surface_preset balanced --visualizer newton_gl

The material script has presets for stiffness, compressibility, friction,
yielding, hardening, dilatancy, viscosity, and particle jitter. Use
``--variant_index`` to run a single specimen. The G1 tutorial downloads a
policy checkpoint unless you provide ``--checkpoint``; it requires a CUDA
device and the ``rsl-rl`` extra. Newton RTX additionally requires
``--extra ovrtx``.

Read the `tutorial README <https://github.com/isaac-sim/IsaacLab/blob/develop/scripts/tutorials/08_mpm/README.md>`_
for the controlled settings and interpretation limits. For scene authoring
and solver configuration, see :ref:`newton-using-mpm`.
