:orphan:

.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _mpm-tutorials:

Exploring MPM materials and coupling
====================================

The Material Point Method (MPM) represents materials with particles that can
deform, compact, and flow. Stiffness, yielding, friction, and viscosity
determine how these materials respond to forces and collisions.

In this tutorial, we compare material parameters in simple impact scenes,
explore the nearly rigid limit, and reconstruct a fluid surface from particles.
We then use a walking G1 robot to see how particle reaction forces affect
rigid-body motion. Each comparison helps isolate a different part of the
simulation so you can adapt the settings to your own scene.

For an introduction to spawning MPM objects and configuring the Newton solver,
see :ref:`newton-using-mpm`. The comparisons below illustrate qualitative
behavior; they are starting points for experiments rather than calibrated
material models.


Running the tutorials
---------------------

The material and rigid-body tutorials launch Kit by default, requiring the
``isaacsim`` extra. The surface-reconstruction tutorial launches Newton GL by
default. Use ``--viz`` to select another visualizer; Newton RTX also requires
``uv run --extra ovrtx``.


Comparing material parameters
-----------------------------

The ``material_parameters.py`` tutorial shows how one parameter changes a
material's response to impact. The stiffness and compressibility presets drop
spheres onto plates. The other presets drop blocks onto horizontal cylinders,
where you can observe spreading, fragmentation, and the shape left after impact.

Elastic stiffness with ``--press_spheres``:

.. raw:: html

   <video autoplay loop muted playsinline preload="metadata" aria-label="MPM elastic stiffness comparison under compression" style="width:100%; max-width:960px;">
     <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/mpm_elastic_stiffness.mp4" type="video/mp4">
   </video>

Start with the elastic stiffness comparison:

.. code-block:: bash

   uv run --extra isaacsim python scripts/tutorials/08_mpm/material_parameters.py \
     --preset young_modulus --viz kit

The three specimens share the same geometry, initial velocities, particle
mass, solver settings, and seeded particle layout. Their Young's moduli are
100 kPa, 300 kPa, and 1 MPa. Compare how much each sphere compresses and how it
recovers after impact. Yielding is disabled in this preset to isolate the
elastic response.

Each preset defines the varied field and the values that remain fixed.
``material_values_for_variant()`` combines these with the common material
settings before passing them to ``MPMParticleMaterialCfg``:

.. literalinclude:: ../../../scripts/tutorials/08_mpm/material_parameters.py
   :language: python
   :pyobject: material_values_for_variant

Choose a preset according to the behavior you want to explore:

.. list-table:: Material comparisons
   :header-rows: 1
   :widths: 35 65

   * - ``--preset``
     - Behavior to inspect
   * - ``young_modulus``
     - Elastic compression and recovery
   * - ``poisson_ratio``
     - Volume change under compression
   * - ``friction``
     - Granular spreading and the final angle of repose
   * - ``tensile_yield_ratio``
     - Fragmentation and tensile cohesion
   * - ``yield_pressure``
     - Irreversible compression
   * - ``hardening``
     - Strength gained after plastic compaction
   * - ``dilatancy``
     - Volume change under shear
   * - ``yield_stress``
     - Cohesive flow and retained shape
   * - ``viscosity``
     - Rate-dependent plastic flow
   * - ``particle_jitter``
     - The effect of aligned versus irregular particle packing

The default solver voxel size is 25 mm, with two particles per voxel axis
(eight per interior voxel). Material comparisons reuse the same random offsets
to avoid confusing a change in packing with a change in material response.
Only ``particle_jitter`` compares aligned particles with jittered particles.
Use ``--variant_index 0``, ``1``, or ``2`` to inspect a single specimen; the
jitter preset has two variants.

To apply sustained compression, add ``--press_spheres`` to either sphere preset:

.. code-block:: bash

   uv run --extra isaacsim python scripts/tutorials/08_mpm/material_parameters.py \
     --preset poisson_ratio --press_spheres --viz kit

The plates descend after a two-second settling period and hold the spheres
under compression. Compare the lateral expansion and volume response while
keeping Young's modulus fixed. Let the simulation continue after loading so
you can distinguish transient motion from the settled shape.

See :ref:`newton-tuning-mpm` for the preset values and guidance on checking
resolution, timestep, and convergence before interpreting material behavior.


Approaching the rigid-body limit
--------------------------------

The ``rigid_body_equivalence.py`` tutorial compares spheres, cubes, and capsules
on two identical inclines. MJWarp simulates rigid bodies on the left; Newton
MPM simulates particle versions on the right. Corresponding objects share
their color, initial placement relative to the ramp, outer dimensions, and
total mass.

.. raw:: html

   <video autoplay loop muted playsinline preload="metadata" aria-label="Rigid bodies and MPM objects descending identical inclines" style="width:100%; max-width:960px;">
     <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/mpm_rigid_limit.mp4" type="video/mp4">
   </video>

.. code-block:: bash

   uv run --extra isaacsim python scripts/tutorials/08_mpm/rigid_body_equivalence.py \
     --viz kit

The MPM objects use a large finite stiffness and pressure-yield threshold to
approximate a rigid material. Compare their rolling, sliding, and shape
retention with the rigid bodies. MPM remains a discretized continuum, so a
large stiffness does not guarantee exact rigid-body motion: grid resolution,
particle sampling, timestep, and solver convergence still matter.

Use ``--mpm_young_modulus`` and ``--mpm_yield_pressure`` to explore the transition
to a softer or yielding material. Keep ``--voxel_size``, ``--mpm_substeps``,
and ``--solver_iterations`` fixed when comparing material values, then vary
them separately to check the numerical approximation.


Reconstructing a fluid surface
------------------------------

The ``surface_reconstruction.py`` tutorial drops a water blob into a shallow
pool. It shows how to turn the particle positions into a continuous surface
for rendering, while leaving the particle simulation unchanged.

.. raw:: html

   <video autoplay loop muted playsinline preload="metadata" aria-label="Reconstructed water surface falling into a shallow pool" style="width:100%; max-width:960px;">
     <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/mpm_surface_reconstruction.mp4" type="video/mp4">
   </video>

.. code-block:: bash

   uv run python scripts/tutorials/08_mpm/surface_reconstruction.py \
     --surface_preset balanced --viz newton_gl

Run the same scene again with another ``--surface_preset``:

.. list-table:: Surface comparisons
   :header-rows: 1
   :widths: 35 65

   * - Preset
     - Change relative to ``balanced``
   * - ``balanced``
     - 25 mm reconstruction grid, anisotropic kernels, one mesh-smoothing pass
   * - ``coarse_grid``
     - 50 mm reconstruction grid
   * - ``isotropic``
     - Isotropic kernels
   * - ``heavy_smoothing``
     - Six mesh-smoothing passes

Look at the splash sheets, small surface features, and the pool as it settles.
Use ``--fluid_render_mode both`` to inspect the mesh alongside the source
particles. The reconstruction grid is separate from the MPM solver grid;
changing it affects surface detail and extraction cost rather than particle
motion. Individual ``--surface_*`` options override the selected preset.

The tutorial constructs ``newton.geometry.ParticleSurface`` after
``sim.reset()`` and passes it to ``ParticleSurfaceRenderer``. The simulation
loop calls the renderer's ``update()`` before ``sim.render()`` so the viewers
receive the mesh for the current particle state. See
:ref:`newton-using-mpm` for this integration pattern.

.. note::

   Surface reconstruction supports Newton GL and Newton RTX only, not Kit,
   Viser, or Rerun. Select ``--fluid_render_mode particles`` to use these
   visualizers without surface reconstruction.


Comparing one-way and two-way coupling
--------------------------------------

The ``mpm-g1-coupling`` example uses a trained G1 walking policy to cross shallow
sand, snow, and clay strips on a rigid runway. The robot starts on the bare
runway before reaching the particles. This comparison shows how material
resistance affects a controller trained for rigid terrain.

Run the two-way comparison from a source checkout:

.. code-block:: bash

   uv run --extra isaacsim --extra rsl-rl isaaclab example mpm-g1-coupling \
     --coupling two_way --viz kit

Then repeat it with ``--coupling one_way``. Both modes let the robot displace
particles. Two-way coupling returns particle reaction forces to the robot;
one-way coupling sets proxy feedback relaxation to zero from the initial state.
Compare the robot's progress and balance as well as the displaced particles.
The policy may slow down, become stuck, or fall when it encounters the material.

By default, the example exposes the robot's full collision geometry to MPM.
Use ``--proxy_bodies feet`` or ``--proxy_bodies lower_legs`` to explore how the
choice of coupled geometry affects interaction. Both ``--proxy_mode`` choices,
``lagged`` and ``staggered``, support feedback; the scheduling mode alone does
not select one-way coupling. See :ref:`newton-coupled-solvers` for the proxy
configuration and feedback controls.

The example requires CUDA and downloads the published
``Isaac-Velocity-Flat-G1`` policy unless you supply ``--checkpoint``. It is also
available in the installed package:

.. code-block:: bash

   uvx --from 'isaaclab[isaacsim,rsl-rl]' isaaclab example mpm-g1-coupling \
     --coupling two_way --viz kit
