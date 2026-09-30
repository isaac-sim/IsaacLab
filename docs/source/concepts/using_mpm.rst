.. _newton-using-mpm:

Using Implicit MPM
==================

Newton's implicit Material Point Method (MPM) solver models particle materials
such as granular media. MPM support and rigid-MPM coupling are experimental.
Start with the compact ``mpm-granular`` example; the ``snowball-smash`` and
``teapot-fill`` demos provide polished coupling and cavity-sampling showcases.

The launcher uses ``cuda:0`` by default. Pass ``--device cuda:N`` to select a
different GPU.


Explore MPM Scenes
------------------

These examples and demos introduce MPM construction and interaction before any
parameter study. They default to Newton GL for interactive viewing:

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Granular drop

      .. raw:: html

         <video autoplay loop muted playsinline controls preload="metadata" style="width:100%;">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/mpm_granular.mp4" type="video/mp4">
         </video>

      .. code-block:: bash

         uv run isaaclab example mpm-granular

   .. grid-item-card:: Two-way sphere pit

      .. raw:: html

         <video autoplay loop muted playsinline controls preload="metadata" style="width:100%;">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/mpm_two_way.mp4" type="video/mp4">
         </video>

      .. code-block:: bash

         uv run isaaclab example mpm-two-way-coupling

   .. grid-item-card:: Snowball smash

      .. raw:: html

         <video autoplay loop muted playsinline controls preload="metadata" style="width:100%;">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/mpm_snowball.mp4" type="video/mp4">
         </video>

      .. code-block:: bash

         uv run isaaclab demo snowball-smash

   .. grid-item-card:: Teapot fill

      .. raw:: html

         <video autoplay loop muted playsinline controls preload="metadata" style="width:100%;">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/mpm_teapot_particles.mp4" type="video/mp4">
         </video>

      .. code-block:: bash

         uv run isaaclab demo teapot-fill --fluid_render_mode particles

.. _franka-pour-reset-artifact:

Train and Regenerate Franka Pour
---------------------------------

The ``IsaacContrib-Franka-Pour`` task restores episodes from a reset artifact
containing the connected 14-phase reset distribution. The canonical 20,000-row
artifact downloads from the standard Isaac Lab asset root on first use, so
training needs no artifact setup:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Franka-Pour \
     --num_envs 2048

The checked-in generator remains the executable reference for reproducing or
customizing the distribution. It takes about two minutes on an L40S-class GPU
and writes a local artifact that can be selected explicitly:

.. code-block:: bash

   uv run python scripts/tools/generate_franka_pour_reset_dataset.py
   uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Franka-Pour \
     --num_envs 2048 \
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

   uv run --extra isaacsim isaaclab play --rl_library rsl_rl --task IsaacContrib-Franka-Pour \
     --checkpoint /path/to/model.pt --num_envs 1 --visualizer kit


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

Both :class:`~isaaclab_newton.sim.MPMGridCfg` and
:class:`~isaaclab_newton.sim.MPMPointsCfg` author schema-valid
``UsdGeom.Points`` simulation geometry. Grid configurations still generate the
lattice in Isaac Lab before authoring explicit points, widths, velocities, and
masses. Materials use ``NewtonMPMMaterialAPI``; damping [s] is authored as
``damping * young_modulus`` [Pa·s]. ``critical_fraction`` remains solver-global
on :class:`~isaaclab_newton.physics.MPMSolverCfg` and is authored on the owning
``NewtonMPMSceneAPI`` physics scene.

Grid jitter is generated once in asset-local coordinates with a fixed seed, so
USD clones share the same local particle distribution. Use reset events or domain
randomization when each environment needs an independent distribution.


Controlled material experiments, comparison demos, and practical solver advice
are collected in :ref:`newton-tuning-mpm`.


Render a Particle Surface
-------------------------

MPM simulation state remains a set of particles. Surface reconstruction is an
optional visualization pass: it does not change particle motion, collisions, or
material behavior. Run the teapot demo to compare the available modes:

.. code-block:: bash

   # Reconstructed surface
   uv run isaaclab demo teapot-fill \
     --visualizer newton_gl --fluid_render_mode surface
   # Surface and source particles together
   uv run isaaclab demo teapot-fill \
     --visualizer newton_gl --fluid_render_mode both
   # Path-traced translucent surface
   uv run --extra ovrtx isaaclab demo teapot-fill \
     --visualizer newton_rtx --fluid_render_mode surface

The teapot demo defaults to the reconstructed surface. Select ``particles`` or
``both`` to inspect the source MPM state in Newton GL or Newton RTX.

Surface rendering is available in the Newton GL and Newton RTX visualizers.
The Kit visualizer continues to render the MPM particles directly.

.. raw:: html

   <video autoplay loop muted playsinline controls preload="metadata" style="width:100%; max-width:960px;">
     <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/mpm_surface_reconstruction.mp4" type="video/mp4">
   </video>

This recording demonstrates rendering appearance, not a different
constitutive model.

The teapot demo configures ``newton.geometry.ParticleSurface`` after
``sim.reset()`` and updates the mesh before ``sim.render()``. Its renderer
captures extraction on CUDA when enabled and stages the dynamic mesh inside
the viewer's frame lifecycle.

Tune reconstruction independently from the simulation:

* ``voxel_size`` controls surface detail and memory use. It can be smaller than
  the MPM solver voxel size.
* ``kernel_radius`` controls how far each particle contributes to the surface.
  Start near three times the particle spacing.
* ``max_grid_cells`` provides fixed-capacity storage for CUDA graph capture.
  Increase it if the reconstructed domain outgrows the reserved grid.
* Anisotropic kernels preserve sheets and stretched fluid features better, but
  cost more than isotropic kernels.

The demo uses a reusable helper to handle CUDA graph capture, empty surfaces,
inactive particles, and dynamic topology:

.. dropdown:: ``ParticleSurfaceRenderer`` implementation
   :icon: code

   .. literalinclude:: ../../../source/isaaclab_visualizers/isaaclab_visualizers/newton/particle_surface.py
      :language: python
      :pyobject: ParticleSurfaceRenderer


Next Steps
----------

See :ref:`newton-tuning-mpm` for resolution and convergence tuning, controlled
material comparisons, and the nearly rigid MPM limit. See
:ref:`newton-coupled-solvers` for rigid--MPM coupling.
