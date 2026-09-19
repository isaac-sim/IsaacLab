.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _newton-tuning-mpm:

MPM Solver and Material Tuning
==============================

Start with :ref:`newton-using-mpm` for scene construction. These experiments
help separate numerical accuracy, constitutive response, coupling, and surface
appearance. They are qualitative examples, not calibrated material models or
performance benchmarks.

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


Tune Rigid-MPM Coupling
-----------------------

For :class:`~isaaclab_contrib.coupling.CouplerProxyCfg`, first stabilize each
solver alone. Then tune the additional controls:

* ``CouplerEntryCfg.substeps`` divides one coupled step for that entry. Increase
  the MPM entry's value when only the particle solve needs a smaller timestep.
* ``CouplerProxyCfg.iterations`` repeats the proxy exchange and relaxation; it
  does not replace smaller physical timesteps.
* ``CouplerProxyMappingCfg.mass_scale`` scales the source body's effective mass
  and inertia only in the destination proxy view. It does not change the body's
  authored mass in the rigid solver.
* ``CouplerProxyMappingCfg.proxy_relaxation`` relaxes updates to the force fed
  back to the source. With fixed relaxation, ``0`` preserves the initial zero
  feedback for a one-way run; ``1`` accepts each new force estimate. Values
  between them blend the new estimate with the previous feedback. Start each
  comparison from a fresh state rather than switching an ongoing run to zero.
  ``proxy_relaxation_mode="aitken"`` adapts the value within a coupled step and
  clamps it between ``proxy_relaxation_min`` and
  ``proxy_relaxation_max``.

Start ``mass_scale`` at ``1`` for a freely moving collider. Increase it when the
rigid solver strongly constrains the collider during MPM contact. For example, a
cup resting on a table has much greater effective resistance in the supported
direction than its free-body mass suggests. Sweep finite values geometrically,
such as ``1``, ``10``, and ``100``, and keep the smallest value that prevents
unrealistic proxy motion. Newton requires a finite positive value: do not use
infinity. An excessively large scalar also suppresses legitimate motion in
unsupported directions and can make the interaction effectively one-way.

Do not use ``mode="lagged"`` as a synonym for one-way coupling. Both ``lagged``
and ``staggered`` transfer modes can return forces. Use zero
``proxy_relaxation`` when the intended experiment must suppress feedback.

Validate both the supported and free-moving cases after changing coupling. If
the uncoupled systems are unstable, fix their timestep, contacts, and reset
states before adjusting ``mass_scale`` or coupling iterations.


Run Controlled Tuning Experiments
---------------------------------

The examples in ``scripts/demos/mpm/tuning`` are controlled experiments rather
than general-purpose scenes. They keep the geometry, initial particles, camera,
and solver configuration fixed while changing one quantity. They live beside
the other standalone MPM demos because each file is directly executable and
owns a complete scene; reusable solver and material APIs remain in
``isaaclab_newton``.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Material response

      Compare up to three specimens while changing one constitutive parameter.

      .. code-block:: bash

         uv run python scripts/demos/mpm/tuning/material_parameters.py \
           --preset young_modulus --visualizer kit

   .. grid-item-card:: Rigid-body limit

      Compare matched MJWarp rigid primitives with nearly rigid MPM particles.

      .. code-block:: bash

         uv run python scripts/demos/mpm/tuning/rigid_body_equivalence.py \
           --visualizer kit

   .. grid-item-card:: One-way and two-way coupling

      Deploy the published G1 policy across successive sand, snow, and clay
      strips. The one-way run moves particles without returning reaction forces.

      .. code-block:: bash

         uv run --extra rsl-rl python scripts/demos/mpm/tuning/g1_coupling.py \
           --coupling two_way --visualizer kit

   .. grid-item-card:: Surface reconstruction

      Keep one water simulation fixed while changing only reconstruction
      resolution, kernel anisotropy, or smoothing.

      .. code-block:: bash

         uv run python scripts/demos/mpm/tuning/surface_reconstruction.py \
           --surface_preset balanced --visualizer newton_gl

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
     - 10 kPa, 100 kPa, 1 MPa
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
* Use deterministic jitter for constitutive comparisons so lattice alignment
  does not dominate the motion. Keep the no-jitter case as a separate packing
  diagnostic.
* Run until impact, peak deformation, recovery or flow, and the final settled
  state are all visible. A short clip can make different materials appear
  identical.
* Tune numerical resolution and timestep before interpreting material values.
  Record the voxel size, particles per voxel axis, particle count, physics
  timestep, substeps, iterations, tolerance, random seed, and code revision.
* Treat surface reconstruction as rendering. Keep the MPM state identical when
  comparing surface parameters, and do not infer a material change from a
  smoother reconstructed mesh.
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
  is not an independent sweep of volume stiffness. The preset uses 20 kPa
  Young's modulus so the impact produces visible strain.
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
* **G1 coupling:** hold the policy, seed, commands, collision proxies, and
  material strips fixed between runs. A fall or stall is a legitimate result
  of feedback, not a success metric. The policy was trained on rigid ground,
  and this demonstration is not a controlled policy-performance benchmark.
* **Surface reconstruction:** run identical water dynamics and vary only the
  extraction parameters. A smoother mesh is not evidence of a more viscous
  material. Kit shows the particle baseline; surface meshes require Newton GL
  or Newton RTX.


Example Recordings
------------------

.. note::

   Selected recordings are awaiting publication to the documentation media host.
   The slots below deliberately have no embedded player until the public files
   are available. Media are hosted outside Git, following the other deformable
   tuning examples. Publication filenames are versioned to avoid replacing
   assets already used in presentations.

Material Response
~~~~~~~~~~~~~~~~~

The selected material clips were recorded in Kit at 1920 × 1080. Their
annotations belong to those recordings; use the CLI and recorded configuration
to reproduce a comparison rather than treating video duration as simulation time.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Elastic stiffness

      Compression and rebound at 10 kPa, 100 kPa, and 1 MPa.

      Run ``material_parameters.py --preset young_modulus --visualizer kit``.

      Video pending: ``mpm_tune_young_modulus_20260919.mp4``.

   .. grid-item-card:: Compressibility

      Poisson ratios 0.05, 0.30, and 0.499 at E = 20 kPa.

      Run ``material_parameters.py --preset poisson_ratio --visualizer kit``.

      Video pending: ``mpm_tune_poisson_ratio_20260919.mp4``.

   .. grid-item-card:: Granular friction

      Runout for internal friction 0, 0.68, and 2.0; cohesion is unchanged.

      Run ``material_parameters.py --preset friction --visualizer kit``.

      Video pending: ``mpm_tune_friction_20260919.mp4``.

   .. grid-item-card:: Tensile cohesion

      Tensile yield ratios 0, 0.01, and 0.05.

      Run ``material_parameters.py --preset tensile_yield_ratio --visualizer kit``.

      Video pending: ``mpm_tune_tensile_yield_ratio_20260919.mp4``.

   .. grid-item-card:: Pressure yielding

      Stronger impact reveals compression at 100 kPa, 1 MPa, and 4 MPa.

      Run ``material_parameters.py --preset yield_pressure --visualizer kit``.

      Video pending: ``mpm_tune_yield_pressure_20260919.mp4``.

   .. grid-item-card:: Plastic hardening

      Fixed strength (0) versus slow (0.05) and rapid (5) strength buildup.

      Run ``material_parameters.py --preset hardening --visualizer kit``.

      Video pending: ``mpm_tune_hardening_20260919.mp4``.

   .. grid-item-card:: Shear dilatancy

      Spreading and packing for dilatancy 0, 0.1, and 1.

      Run ``material_parameters.py --preset dilatancy --visualizer kit``.

      Video pending: ``mpm_tune_dilatancy_20260919.mp4``.

   .. grid-item-card:: Cohesive yield stress

      Retained shape at 0, 10 kPa, and 20 kPa.

      Run ``material_parameters.py --preset yield_stress --visualizer kit``.

      Video pending: ``mpm_tune_yield_stress_20260919.mp4``.

   .. grid-item-card:: Plastic viscosity

      Rate of plastic flow at 0, 10 Pa·s, and 500 Pa·s.

      Run ``material_parameters.py --preset viscosity --visualizer kit``.

      Video pending: ``mpm_tune_viscosity_20260919.mp4``.

   .. grid-item-card:: Initial packing

      Aligned sampling versus deterministic 30% particle-spacing jitter.

      Run ``material_parameters.py --preset particle_jitter --visualizer kit``.

      Video pending: ``mpm_tune_particle_jitter_20260919.mp4``.

Comparison Demos
~~~~~~~~~~~~~~~~

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Nearly rigid MPM versus MJWarp

      Inspect matched colored primitives rolling down separated inclines. This
      older presentation recording is not an accuracy claim for the current
      implementation; high stiffness does not enforce rigid shape constraints.

      Video pending: ``mpm_rigid_equivalence_kit_wide18_y26_20260919.mp4``.

   .. grid-item-card:: G1 coupling examples

      Historical recordings use 25 mm voxels and 20 cm strips; the current demo
      defaults to 40 mm voxels and 16 cm strips for faster iteration. The one-way
      recording uses lower-leg proxies, whereas the full-body two-way recording
      uses all robot collision geometry. Do not present those two as a
      single-variable comparison. Rerun both modes with identical
      ``--proxy_bodies`` settings for that purpose.

      Video pending: ``g1_mpm_one_way_kit_20260919.mp4``.

      Video pending: ``g1_mpm_two_way_all_geometry_kit_20260919.mp4``.

   .. grid-item-card:: Water surface reconstruction

      A Newton RTX recording of the balanced water-surface preset. This shows
      reconstruction appearance, not a different MPM constitutive model.

      Video pending: ``mpm_surface_splash_balanced_rtx_20260919.mp4``.

   .. grid-item-card:: Core MPM scenes

      Kit particle recordings of the granular drop, two-way sphere pit,
      snowball smash, and teapot fill are separate from parameter studies.

      Videos pending: ``mpm_granular_kit_20260919.mp4``,
      ``mpm_two_way_coupling_kit_20260919.mp4``,
      ``mpm_snowball_smash_kit_20260919.mp4``, and
      ``mpm_teapot_fill_particles_kit_20260919.mp4``.
