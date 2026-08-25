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

   uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Franka-Pour --num_envs 2048 --device cuda:0

The checked-in generator remains the executable reference for reproducing or
customizing the distribution. It takes about two minutes on an L40S-class GPU
and writes a local artifact that can be selected explicitly:

.. code-block:: bash

   uv run python scripts/tools/generate_franka_pour_reset_dataset.py --device cuda:0
   uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Franka-Pour --num_envs 2048 --device cuda:0 env.reset_dataset_path=datasets/franka_pour/reset_dataset.pt

The task validates the payload's stored content digest automatically. Setting
``ISAACSIM_ASSET_ROOT`` redirects the canonical artifact to a compatible local
or self-hosted asset tree. Digest pinning remains available for custom
reproducible experiments.


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
                project_outside_colliders=True,
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
:class:`~isaaclab_newton.sim.MPMParticleMaterialCfg`. Enable
``project_outside_colliders`` for contact scenes when particles otherwise drift
inside colliders; leave it disabled in collider-free scenes to avoid the extra
projection pass.


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
