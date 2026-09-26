.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _newton-tuning-mpm:

MPM Solver
==========

Start with :ref:`newton-using-mpm` for scene construction. These experiments
help separate numerical accuracy from constitutive response. They are
qualitative examples, not calibrated material models or performance benchmarks.

Tune Resolution, Time, Then Convergence
---------------------------------------

Tune one group at a time in this order:

1. **Voxel and particle resolution.** ``MPMSolverCfg.voxel_size`` controls the
   background grid. Smaller voxels resolve thinner geometry but increase active
   cells and memory. ``MPMGridCfg.particles_per_cell`` controls particle density;
   doubling it along each axis creates about eight times as many particles in
   3D. Start coarse, then refine until the measured behavior stops changing.
2. **Timestep and substeps.** Each Newton substep uses
   ``SimulationCfg.dt / NewtonCfg.num_substeps``. Reduce ``dt`` or increase
   ``num_substeps`` first when contacts tunnel, jitter, or become unstable.
   Substeps do not change the policy period, which also includes environment
   decimation.
3. **Iterations and tolerance.** ``MPMSolverCfg.max_iterations`` caps the
   rheology solve; ``tolerance`` permits an earlier exit after convergence.
   Increase the cap only when the solver reaches it, and lower the tolerance
   only when tighter convergence improves a physical metric. These settings do
   not repair an unstable timestep, invalid reset, or incorrect collider.


Run Controlled Tuning Experiments
---------------------------------

The ``mpm-*`` tuning examples are controlled experiments rather than
general-purpose scenes. They keep the geometry, initial particles, camera,
and solver configuration fixed while changing one quantity. Run them from a
source checkout with ``uv run isaaclab example <name>`` or from an installed
wheel with ``uvx isaaclab example <name>``. Reusable solver and material APIs
remain in ``isaaclab_newton``.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Material response

      Compare up to three specimens while changing one constitutive parameter.

      .. code-block:: bash

         uv run isaaclab example mpm-material-tuning \
           --preset young_modulus --visualizer kit

   .. grid-item-card:: Rigid-body limit

      Compare matched MJWarp rigid primitives with nearly rigid MPM particles.

      .. code-block:: bash

         uv run isaaclab example mpm-rigid-equivalence \
           --visualizer kit

The material runner includes the following presentation presets. Use
``--variant_index`` to render one member of a comparison, or omit it for the
side-by-side view.

.. list-table:: Material tuning presets
   :header-rows: 1
   :widths: 21 35 44

   * - Preset
     - Varied values
     - Behavior to inspect
   * - ``young_modulus``
     - 100 kPa, 300 kPa, 1 MPa
     - Compression, recovery, and impact rebound
   * - ``poisson_ratio``
     - 0.05, 0.30, 0.499
     - Volume loss versus near-incompressibility
   * - ``friction``
     - 0, 0.68, 2.0
     - Runout and the final angle of repose
   * - ``tensile_yield_ratio``
     - 0, 0.01, 0.05
     - Fragmentation versus tensile cohesion
   * - ``yield_pressure``
     - 100 kPa, 1 MPa, 4 MPa
     - Onset and amount of irreversible compression
   * - ``hardening``
     - 0, 0.05, 5.0
     - Strength gained after plastic compaction
   * - ``dilatancy``
     - 0, 0.1, 1.0
     - Compaction versus expansion under shear
   * - ``yield_stress``
     - 0, 10 kPa, 20 kPa
     - Cohesive flow and retained shape
   * - ``viscosity``
     - 0, 10 Pa·s, 500 Pa·s
     - Rate-dependent plastic flow
   * - ``particle_jitter``
     - 0%, 30% of particle spacing
     - Grid-alignment artifacts and packing symmetry


Design a Useful Parameter Study
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use these practices when adapting the examples:

* Change one physical quantity at a time. Use the same generated positions,
  particle mass, collider, timestep, and camera for every variant.
* Choose values that bracket visibly different regimes. Geometric spacing is
  usually more informative than small linear increments for stiffness, yield,
  viscosity, and coupling mass scales.
* Preserve the feature the comparison is meant to explain. If a soft elastic
  specimen loses its silhouette before rebound can be read, reduce the impact
  energy or raise the lower end of the stiffness range. Do not shorten a clip
  merely to hide later breakup.
* Use deterministic jitter for constitutive comparisons so lattice alignment
  does not dominate the motion. Keep the no-jitter case as a separate packing
  diagnostic.
* Run until impact, peak deformation, recovery or flow, and the final settled
  state are all visible. A short clip can make different materials appear
  identical.
* Tune numerical resolution and timestep before interpreting material values.
  Record the voxel size, particles per voxel axis, particle count, physics
  timestep, substeps, iterations, tolerance, random seed, and code revision.
* Check a quantitative signal alongside the video when possible: center of
  mass, runout distance, rebound height, retained volume, or settled height.

``particles_per_cell`` is the number of particles along one voxel axis. A value
of ``2`` therefore produces approximately eight particles per filled voxel in
3D, not two. Refining both voxel size and particle density can increase memory
and runtime rapidly.


Interpret the Comparisons
-------------------------

* **Elastic response:** Poisson's ratio changes both bulk and shear moduli at
  fixed Young's modulus. Values close to 0.5 approach incompressibility; this
  is not an independent sweep of volume stiffness. The preset uses 50 kPa
  Young's modulus and a gentler drop so the impact produces visible strain
  without obscuring the comparison through loss of the specimen silhouette.
* **Plastic response:** yielding, hardening, friction, and dilatancy interact.
  Keep the other fields fixed within a preset, not necessarily between presets.
  A high-friction label is not a water-content model; these sand, snow, and
  clay labels describe illustrative responses rather than measured materials.
* **Nearly rigid MPM:** high stiffness and yield limits suppress intended strain
  and plastic flow but do not impose rigid-body constraints. Residual shape
  drift, contact differences, and particle sampling error can remain. Compare
  silhouettes and mass as well as motion; increase resolution only after
  checking timestep and convergence. The default three-particle-per-axis
  comparison is intentionally more expensive than the other examples.


Example Recordings
------------------

Recordings are hosted outside Git on the Isaac Lab documentation media host,
following the other deformable tuning examples.

Material Response
~~~~~~~~~~~~~~~~~

The selected material clips were recorded in Kit at 1920 × 1080. Their
annotations belong to those recordings; use the CLI and recorded configuration
to reproduce a comparison rather than treating video duration as simulation time.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Elastic stiffness

      .. raw:: html

         <video autoplay loop muted playsinline controls preload="metadata" style="width:100%;">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/mpm_elastic_stiffness.mp4" type="video/mp4">
         </video>

      Plate compression at 100 kPa, 300 kPa, and 1 MPa.

      Run ``isaaclab example mpm-material-tuning --preset young_modulus --press_spheres --visualizer kit``.

   .. grid-item-card:: Compressibility

      .. raw:: html

         <video autoplay loop muted playsinline controls preload="metadata" style="width:100%;">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/mpm_compressibility.mp4" type="video/mp4">
         </video>

      Plate compression at Poisson ratios 0.05, 0.30, and 0.499 with E = 50 kPa.

      Run ``isaaclab example mpm-material-tuning --preset poisson_ratio --press_spheres --visualizer kit``.

   .. grid-item-card:: Granular friction

      .. raw:: html

         <video autoplay loop muted playsinline controls preload="metadata" style="width:100%;">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/mpm_friction.mp4" type="video/mp4">
         </video>

      Runout for internal friction 0, 0.68, and 2.0; cohesion is unchanged.

      Run ``isaaclab example mpm-material-tuning --preset friction --visualizer kit``.

   .. grid-item-card:: Tensile cohesion

      .. raw:: html

         <video autoplay loop muted playsinline controls preload="metadata" style="width:100%;">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/mpm_tensile_strength.mp4" type="video/mp4">
         </video>

      Tensile yield ratios 0, 0.01, and 0.05.

      Run ``isaaclab example mpm-material-tuning --preset tensile_yield_ratio --visualizer kit``.

   .. grid-item-card:: Pressure yielding

      .. raw:: html

         <video autoplay loop muted playsinline controls preload="metadata" style="width:100%;">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/mpm_pressure_yield.mp4" type="video/mp4">
         </video>

      Stronger impact reveals compression at 100 kPa, 1 MPa, and 4 MPa.

      Run ``isaaclab example mpm-material-tuning --preset yield_pressure --visualizer kit``.

   .. grid-item-card:: Plastic hardening

      .. raw:: html

         <video autoplay loop muted playsinline controls preload="metadata" style="width:100%;">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/mpm_hardening.mp4" type="video/mp4">
         </video>

      Fixed strength (0) versus slow (0.05) and rapid (5) strength buildup.

      Run ``isaaclab example mpm-material-tuning --preset hardening --visualizer kit``.

   .. grid-item-card:: Shear dilatancy

      .. raw:: html

         <video autoplay loop muted playsinline controls preload="metadata" style="width:100%;">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/mpm_dilatancy.mp4" type="video/mp4">
         </video>

      Spreading and packing for dilatancy 0, 0.1, and 1.

      Run ``isaaclab example mpm-material-tuning --preset dilatancy --visualizer kit``.

   .. grid-item-card:: Cohesive yield stress

      .. raw:: html

         <video autoplay loop muted playsinline controls preload="metadata" style="width:100%;">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/mpm_yield_stress.mp4" type="video/mp4">
         </video>

      Retained shape at 0, 10 kPa, and 20 kPa.

      Run ``isaaclab example mpm-material-tuning --preset yield_stress --visualizer kit``.

   .. grid-item-card:: Plastic viscosity

      .. raw:: html

         <video autoplay loop muted playsinline controls preload="metadata" style="width:100%;">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/mpm_viscosity.mp4" type="video/mp4">
         </video>

      Rate of plastic flow at 0, 10 Pa·s, and 500 Pa·s.

      Run ``isaaclab example mpm-material-tuning --preset viscosity --visualizer kit``.

   .. grid-item-card:: Initial packing

      .. raw:: html

         <video autoplay loop muted playsinline controls preload="metadata" style="width:100%;">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/mpm_particle_jitter.mp4" type="video/mp4">
         </video>

      Aligned sampling versus deterministic 30% particle-spacing jitter.

      Run ``isaaclab example mpm-material-tuning --preset particle_jitter --visualizer kit``.

Nearly Rigid Limit
~~~~~~~~~~~~~~~~~~

.. grid:: 1
   :gutter: 2

   .. grid-item-card:: Nearly rigid MPM versus MJWarp

      .. raw:: html

         <video autoplay loop muted playsinline controls preload="metadata" style="width:100%;">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/mpm_rigid_limit.mp4" type="video/mp4">
         </video>

      Inspect matched colored primitives rolling down separated inclines. This
      older presentation recording is not an accuracy claim for the current
      implementation; high stiffness does not enforce rigid shape constraints.
