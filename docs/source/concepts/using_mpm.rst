.. _newton-using-mpm:

Using Implicit MPM
==================

Newton's implicit Material Point Method (MPM) solver models particle materials
such as granular media. MPM support and rigid-MPM coupling are experimental.
Start with the compact ``scripts/demos/mpm/newton_mpm_granular.py`` example;
``snowball_smash.py`` adds coupling and ``teapot_fill.py`` adds cavity sampling.


.. _franka-pour-reset-artifact:

Train and Regenerate Franka Pour
---------------------------------

The ``IsaacContrib-Franka-Pour`` task restores episodes from a reset artifact
containing the connected 14-phase reset distribution. The canonical 20,000-row
artifact downloads from the standard Isaac Lab asset root on first use, so
training needs no artifact setup:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Franka-Pour \
     --num_envs 2048 --device cuda:0

The checked-in generator remains the executable reference for reproducing or
customizing the distribution. It takes about two minutes on an L40S-class GPU
and writes a local artifact that can be selected explicitly:

.. code-block:: bash

   uv run python scripts/tools/generate_franka_pour_reset_dataset.py --device cuda:0
   uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Franka-Pour \
     --num_envs 2048 --device cuda:0 \
     env.reset_dataset_path=datasets/franka_pour/reset_dataset.pt

The task validates the payload's stored content digest automatically. Setting
``ISAACSIM_ASSET_ROOT`` redirects the canonical artifact to a compatible local
or self-hosted asset tree. Digest pinning remains available for custom
reproducible experiments.

The source uses a non-colliding analytic fill volume whose height is controlled
by ``env.source_fill_level`` in ``(0, 1]``. The default ``0.70`` produces a
735-particle jittered lattice up to roughly 70% of the cup height.
``env.pour_target_frac`` independently controls the fraction of that live
payload that must reach the receiver.

Play a checkpoint in Kit with the canonical task configuration; no external
callback or particle override is required:

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl --task IsaacContrib-Franka-Pour \
     --checkpoint /path/to/model.pt --num_envs 1 --device cuda:0 --visualizer kit


Minimal Setup
-------------

Use the same voxel size for the solver grid and particle generator. Add the
generated object to an :class:`~isaaclab.scene.InteractiveSceneCfg` like any
other declarative asset.

.. code-block:: python

    import isaaclab.sim as sim_utils
    from isaaclab_newton.assets import MPMObjectCfg
    from isaaclab_newton.physics import MPMSolverCfg, NewtonCfg
    from isaaclab_newton.sim.spawners.mpm import MPMGridCfg

    voxel_size = 0.02

    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 100.0,
        physics=NewtonCfg(
            solver_cfg=MPMSolverCfg(
                voxel_size=voxel_size,
                max_iterations=100,
                tolerance=1.0e-4,
            ),
            num_substeps=2,
        ),
    )

    media = MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Media",
        spawn=MPMGridCfg(
            lower=(-0.1, -0.1, 0.0),
            upper=(0.1, 0.1, 0.2),
            voxel_size=voxel_size,
            particles_per_cell=2.0,
            particle_placement="cell_center",
        ),
    )

Tune the particle material separately through
:class:`~isaaclab_newton.sim.MPMParticleMaterialCfg`. The implicit solve already
resolves collider contact. ``project_outside_colliders`` adds a hard post-step
correction for particles that remain inside colliders; use it only when that
geometric correction is intentional, and not as a substitute for valid initial
states, collision geometry, or a stable timestep. Coupled MPM entries do not
support this manager-level projection pass.


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


Publish the Demo Videos
~~~~~~~~~~~~~~~~~~~~~~~

Generated videos are intentionally not stored in Git. Upload final MP4 files to
the documentation media host, then replace the cards below with the standard
``raw:: html`` video element used elsewhere in the documentation. Keep these
stable publication names so slide decks and docs can share the same assets.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Video slot — material tuning

      ``mpm_tune_young_modulus.mp4``

      ``mpm_tune_poisson_ratio.mp4``

      ``mpm_tune_friction.mp4``

      ``mpm_tune_tensile_yield_ratio.mp4``

      ``mpm_tune_yield_pressure.mp4``

      ``mpm_tune_hardening.mp4``

      ``mpm_tune_dilatancy.mp4``

      ``mpm_tune_yield_stress.mp4``

      ``mpm_tune_viscosity.mp4``

      ``mpm_tune_particle_jitter.mp4``

   .. grid-item-card:: Video slot — comparison demos

      ``mpm_rigid_body_equivalence.mp4``

      ``mpm_g1_one_way.mp4``

      ``mpm_g1_two_way.mp4``

      ``mpm_surface_reconstruction.mp4``

   .. grid-item-card:: Video slot — core MPM demos

      ``mpm_granular.mp4``

      ``mpm_two_way_coupling.mp4``

      ``mpm_snowball_smash.mp4``

      ``mpm_teapot_fill_particles.mp4``

   .. grid-item-card:: Capture checklist

      Use a fixed camera and resolution, include the full evolution, retain one
      resolved-configuration record per clip, and verify the encoded MP4 on a
      second machine before publishing it.


Render a Particle Surface
-------------------------

MPM simulation state remains a set of particles. Surface reconstruction is an
optional visualization pass: it does not change particle motion, collisions, or
material behavior. Run the teapot example to compare the available modes:

.. code-block:: bash

   # Reconstructed surface
   uv run python scripts/demos/mpm/teapot_fill.py --device cuda:0 \
     --visualizer newton_gl --fluid_render_mode surface
   # Surface and source particles together
   uv run python scripts/demos/mpm/teapot_fill.py --device cuda:0 \
     --visualizer newton_gl --fluid_render_mode both
   # Path-traced translucent surface
   uv run --extra ovrtx python scripts/demos/mpm/teapot_fill.py --device cuda:0 \
     --visualizer newton_rtx --fluid_render_mode surface

The teapot demo defaults to particles. This keeps the source MPM state visible
in Kit and avoids making a reconstruction choice on behalf of the user. Select
``surface`` or ``both`` explicitly for Newton GL or Newton RTX.

Surface rendering is available in the Newton GL and Newton RTX visualizers.
The Kit visualizer continues to render the MPM particles directly.

To reconstruct a surface in another Newton MPM script, create one reusable
``newton.geometry.ParticleSurface`` after ``sim.reset()``. On each render update,
extract from the current Newton particle positions, radii, flags, and world indices,
then pass the returned vertex, triangle-index, and normal arrays to
``NewtonGLVisualizer.log_mesh()`` or ``NewtonRTXVisualizer.log_mesh()`` with
``dynamic=True`` before calling ``sim.render()``. The visualizer stages the latest
mesh by name and submits it inside Newton's required viewer-frame lifecycle.

Tune reconstruction independently from the simulation:

* ``voxel_size`` controls surface detail and memory use. It can be smaller than
  the MPM solver voxel size.
* ``kernel_radius`` controls how far each particle contributes to the surface.
  Start near three times the particle spacing.
* ``max_grid_cells`` provides fixed-capacity storage for CUDA graph capture.
  Increase it if the reconstructed domain outgrows the reserved grid.
* Anisotropic kernels preserve sheets and stretched fluid features better, but
  cost more than isotropic kernels.

The demo handles CUDA graph capture, empty surfaces, inactive particles, and
dynamic topology in one reusable helper:

.. dropdown:: ``FluidSurfaceRenderer`` implementation
   :icon: code

   .. literalinclude:: ../../../scripts/demos/mpm/teapot_fill.py
      :language: python
      :pyobject: FluidSurfaceRenderer


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

Start ``mass_scale`` at ``1`` for a freely moving collider. Increase it when the
rigid solver strongly constrains the collider during MPM contact. For example, a
cup resting on a table has much greater effective resistance in the supported
direction than its free-body mass suggests. Sweep finite values geometrically,
such as ``1``, ``10``, and ``100``, and keep the smallest value that prevents
unrealistic proxy motion. Newton requires a finite positive value: do not use
infinity. An excessively large scalar also suppresses legitimate motion in
unsupported directions and can make the interaction effectively one-way.

Validate both the supported and free-moving cases after changing coupling. If
the uncoupled systems are unstable, fix their timestep, contacts, and reset
states before adjusting ``mass_scale`` or coupling iterations.
