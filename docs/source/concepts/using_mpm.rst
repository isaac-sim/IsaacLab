.. _newton-using-mpm:

Implicit MPM
============

Newton's implicit Material Point Method (MPM) solver models particle materials
such as sand, snow, and other granular media. Isaac Lab integrates it as a
Newton solver configuration, a particle asset type, and an optional coupling
with rigid-body solvers. MPM support and rigid--MPM coupling are experimental.

Start with `Examples and tasks`_ for runnable scenes. See :ref:`newton-tuning-mpm`
for resolution, convergence, material settings, and the nearly rigid MPM limit.


Examples and tasks
------------------

These examples and demos introduce MPM construction and interaction. They
default to Newton GL for interactive viewing. The launcher uses ``cuda:0`` by
default. Pass ``--device cuda:N`` to select a different GPU.

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

Franka Pour
^^^^^^^^^^^

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
by ``env.source_fill_level`` in ``(0, 1]``. The default ``0.70`` fills roughly
70% of the cup height; the particle count follows the requested fill level
while voxel size and particle spacing stay fixed.
``env.pour_target_frac`` independently controls the fraction of that live
payload that must reach the receiver.

Play a checkpoint in Kit with the canonical task configuration; no external
callback or particle override is required:

.. code-block:: bash

   uv run --extra isaacsim isaaclab play --rl_library rsl_rl --task IsaacContrib-Franka-Pour \
     --checkpoint /path/to/model.pt --num_envs 1 --visualizer kit


Coupling with rigid bodies
--------------------------

On its own, the MPM solver advances only particles. Rigid bodies act as
colliders whose motion comes from elsewhere, such as kinematic pose writes, and
the material does not push back on them.

Two-way interaction uses the experimental coupler in
:mod:`isaaclab_contrib.coupling`. A
:class:`~isaaclab_contrib.coupling.CouplerProxyCfg` partitions one Newton model
into named entries:

* a rigid entry, for example MJWarp, that owns the robot or object bodies;
* an MPM entry that owns the particles and must set ``in_place=True``; and
* a proxy mapping that exposes selected rigid bodies as colliders in the MPM
  entry and feeds the resulting impulses back to the rigid entry.

The ``mpm-two-way-coupling`` example, the ``snowball-smash`` demo, and the
Franka Pour task all use this pattern. See :ref:`newton-coupled-solvers` for
proxy and ADMM coupling, exchange modes, and tuning.


Data model
----------

Configuration and USD
^^^^^^^^^^^^^^^^^^^^^

An MPM material is a scene asset. :class:`~isaaclab_newton.assets.MPMObjectCfg`
extends the deformable-object configuration and takes a particle spawner:

* :class:`~isaaclab_newton.sim.spawners.mpm.MPMGridCfg` fills an axis-aligned local box with
  a particle lattice whose resolution follows ``voxel_size`` and
  ``particles_per_cell``.
* :class:`~isaaclab_newton.sim.spawners.mpm.MPMPointsCfg` places particles at explicit
  local positions, with optional velocities, masses, and radii.

Both spawners author schema-valid ``UsdGeom.Points`` simulation geometry. Grid
configurations still generate the lattice in Isaac Lab before authoring
explicit points, widths, velocities, and masses. Per-particle material values
come from :class:`~isaaclab_newton.sim.spawners.mpm.MPMParticleMaterialCfg` and are authored
with ``NewtonMPMMaterialAPI``. Damping [s] is authored as
``damping * young_modulus`` [Pa·s]. Solver-wide settings, including
``critical_fraction``, live on ``MPMSolverCfg``; the schema-representable ones
are authored with ``NewtonMPMSceneAPI`` on the physics scene prim.

MPM objects are cloned through Newton replication. Grid jitter is generated
once in asset-local coordinates with a fixed seed, so USD clones share the same
local particle distribution. Use reset events or domain randomization when each
environment needs an independent distribution.

Use the same voxel size for the solver grid and particle generator. Add the
generated object to an :class:`~isaaclab.scene.InteractiveSceneCfg` like any
other declarative asset:

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

Runtime asset
^^^^^^^^^^^^^

:class:`~isaaclab_newton.assets.MPMObject` implements Isaac Lab's
deformable-object interface, so scene reset, update, and state workflows treat
it like other deformables. Each environment instance holds the same number of
particles, reported by ``particles_per_object``.
:class:`~isaaclab_newton.assets.MPMObjectData` exposes particle positions,
velocities, and combined state in the simulation world frame through
``particle_pos_w``, ``particle_vel_w``, and ``particle_state_w``. The
deformable ``nodal_*`` properties are aliases of these. ``root_pos_w`` and
``root_vel_w`` are per-instance means over the particles. Position, velocity,
and state writers apply immediately. Deformable kinematic targets are not
supported.


Configuration reference
-----------------------

The generated :class:`~isaaclab_newton.physics.MPMSolverCfg` reference is the
source of truth for fields and defaults. The fields group as follows:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Group
     - Fields
     - Effect
   * - Resolution
     - ``voxel_size``, and ``particles_per_cell`` on the spawner
     - Background-grid and particle resolution, which set accuracy, active
       cells, and memory use.
   * - Grid storage
     - ``grid_type``, ``grid_padding``, ``max_active_cell_count``, the
       ``max_*_node_count`` limits, and ``separate_worlds``
     - Sparse, dense, or fixed grid allocation, reserved capacity, and
       whether each world uses its own local grid.
   * - Rheology solve
     - ``max_iterations``, ``tolerance``, ``solver``, and ``warmstart_mode``
     - Iteration cap, convergence tolerance, the solver or warm-start
       sequence, and warm-start source.
   * - Discretization
     - ``transfer_scheme``, ``integration_scheme``, ``velocity_basis``,
       ``strain_basis``, and ``collider_basis``
     - Particle-grid transfer, shape-function support, and the basis
       functions for each field.
   * - Material and background
     - ``critical_fraction`` and ``air_drag``
     - Solver-global yield-surface collapse and numerical drag for empty
       space.
   * - Colliders
     - ``collider_velocity_mode``, ``collider_normal_from_sdf_gradient``,
       and ``project_outside_colliders``
     - Collider velocity and normal computation, and an optional post-substep
       projection of particles out of colliders.

The implicit solve already resolves collider contact. ``project_outside_colliders``
adds a hard post-step correction for particles that remain inside colliders;
use it only when that geometric correction is intentional, and not as a
substitute for valid initial states, collision geometry, or a stable timestep.

CUDA graph capture
^^^^^^^^^^^^^^^^^^

CUDA graph capture of the MPM step is available only when the grid has
capture-stable storage:

* a ``"fixed"`` grid with a positive ``max_active_cell_count``; or
* a ``"sparse"`` grid with a positive ``max_active_cell_count``,
  ``grid_padding=0``, and a ``"Q1"`` or ``"B2"`` velocity basis. The strain basis must
  start with ``"pic"`` or be one of ``"P0"``, ``"P1d"``, ``"Q1d"``, or
  ``"Q1"``. The collider basis must start with ``"pic"`` or be one of
  ``"Q1"``, ``"S2"``, or ``"S3"``.

Dense grids and unbounded grids run without capture.

Resets
^^^^^^

With the MPM solver alone, automatic asset resets leave solver history
untouched, because worlds that share one grid cannot have their history reset
individually. Under a coupler, automatic resets go through the coupled
solver's reset, which passes the per-world reset mask to every entry, including
the MPM entry. Tasks that need an exact history reset, such as Franka Pour, use
independent MPM worlds (``separate_worlds=True``) and call
:func:`~isaaclab_newton.physics.reset_mpm_solver_state` on the physics
manager's backend after writing their complete state. In a coupled run, this also resets the coupled
solver's history.

Try the :ref:`interactive MPM demo <interactive-demo-mpm>` to compare material
references and adjust stiffness, cohesion, friction, and hardening on a
repeatable particle drop.


Rendering a particle surface
----------------------------

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

The teapot demo uses Newton GL and defaults to the reconstructed surface. Select
``particles`` or ``both`` to inspect the source MPM state. Its ``--visualizer`` / ``--viz``
option accepts only ``newton_gl``; omit the option to use that default.

For a path-traced surface, use the surface-reconstruction tutorial:

.. code-block:: bash

   uv run --extra ovrtx python scripts/tutorials/08_mpm/surface_reconstruction.py \
     --visualizer newton_rtx --fluid_render_mode surface

Surface rendering is available in the Kit, Newton GL, and Newton RTX visualizers.

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

To render a surface in your own scene, pass the configured
``newton.geometry.ParticleSurface`` to ``ParticleSurfaceRenderer`` after
``sim.reset()``. Call ``update()`` before ``sim.render()`` to extract and stage
the current mesh. The renderer handles CUDA graph capture, empty surfaces,
inactive particles, and dynamic topology:

.. dropdown:: ``ParticleSurfaceRenderer`` implementation
   :icon: code

   .. literalinclude:: ../../../source/isaaclab_visualizers/isaaclab_visualizers/newton/particle_surface.py
      :language: python
      :pyobject: ParticleSurfaceRenderer


How MPM fits into the Newton backend
------------------------------------

Selecting :class:`~isaaclab_newton.physics.MPMSolverCfg` as the ``solver_cfg`` of
:class:`~isaaclab_newton.physics.NewtonCfg` constructs Newton's ``SolverImplicitMPM``. The solver
represents material as particles and solves its rheology implicitly on a
background voxel grid. The MPM configuration sets up the Newton backend as
follows:

* It advances particles in place on a single Newton state rather than double
  buffering state.
* It resolves collider contact inside the implicit solve. It does not use
  Newton's collision pipeline.
* It treats rigid geometry as colliders. It is not a rigid-body or
  articulation dynamics solver, and it does not accept external rigid-body
  forces. Kinematic bodies are given zero mass before the model is finalized,
  and convex meshes are passed to the solver as triangle meshes.


Limitations
-----------

* Coupled MPM entries must step in place and do not support
  ``project_outside_colliders``.
