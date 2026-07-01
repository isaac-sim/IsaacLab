FeatherPGS Solver
=================

FeatherPGS is an experimental Newton solver for articulated rigid-body systems.
It evaluates reduced-coordinate dynamics and resolves contact and enabled joint
constraints with projected Gauss-Seidel (PGS) iterations. Isaac Lab exposes it
through :class:`~isaaclab_newton.physics.FeatherPGSSolverCfg` and
:class:`~isaaclab_newton.physics.NewtonFeatherPGSManager`.

FeatherPGS uses Newton's :class:`CollisionPipeline`. Configure contact
generation with
:class:`~isaaclab_newton.physics.NewtonCollisionPipelineCfg`, as with the
Featherstone and XPBD Newton managers.

Configuration
-------------

Select FeatherPGS by assigning its solver configuration to
:attr:`~isaaclab_newton.physics.NewtonCfg.solver_cfg`:

.. code-block:: python

    from isaaclab.sim import SimulationCfg
    from isaaclab_newton.physics import (
        FeatherPGSSolverCfg,
        NewtonCfg,
        NewtonCollisionPipelineCfg,
    )

    solver_cfg = FeatherPGSSolverCfg(
        enable_joint_limits=True,
        pgs_iterations=8,
        pgs_mode="split",
        dense_max_constraints=64,
        mf_max_constraints=512,
        pgs_beta=0.05,
        pgs_cfm=1.0e-6,
    )
    newton_cfg = NewtonCfg(
        solver_cfg=solver_cfg,
        collision_cfg=NewtonCollisionPipelineCfg(),
        num_substeps=1,
    )
    sim_cfg = SimulationCfg(dt=1.0 / 200.0, physics=newton_cfg)

The important accuracy and runtime controls are:

* :attr:`~isaaclab_newton.physics.FeatherPGSSolverCfg.pgs_iterations`:
  increases constraint-solve work and usually improves constraint fidelity.
* :attr:`~isaaclab_newton.physics.NewtonCfg.num_substeps`: repeats collision,
  integration, and the solve at a shorter effective time step.
* :attr:`~isaaclab_newton.physics.FeatherPGSSolverCfg.pgs_mode`: selects the
  dense, split, or matrix-free constraint layout.
* :attr:`~isaaclab_newton.physics.FeatherPGSSolverCfg.dense_max_constraints`
  and
  :attr:`~isaaclab_newton.physics.FeatherPGSSolverCfg.mf_max_constraints`:
  preallocate per-world constraint capacity. Undersizing either capacity can
  omit constraints; oversizing it consumes memory and can reduce throughput.
* :attr:`~isaaclab_newton.physics.FeatherPGSSolverCfg.pgs_beta` and
  :attr:`~isaaclab_newton.physics.FeatherPGSSolverCfg.pgs_cfm`: control
  position correction and solver regularization.

Solve Layouts
-------------

The three :attr:`~isaaclab_newton.physics.FeatherPGSSolverCfg.pgs_mode`
values trade memory, kernel work, and feature support:

* ``"dense"`` constructs a full Delassus matrix for all constraints.
* ``"split"`` uses the dense path for articulated contacts and a
  matrix-free path for free rigid bodies.
* ``"matrix_free"`` avoids the full Delassus matrix and recomputes the
  articulated response during PGS iterations.

The automatic kernel selectors use
:attr:`~isaaclab_newton.physics.FeatherPGSSolverCfg.small_dof_threshold` to
choose between loop and tiled implementations. Set
:attr:`~isaaclab_newton.physics.FeatherPGSSolverCfg.nvtx` to ``True`` to
emit solver-stage ranges when profiling.

Current Limitations
-------------------

FeatherPGS support is experimental and is not selected by an Isaac Lab task
preset yet. Task configurations should opt in explicitly and validate their
time step, actuator, contact-capacity, and iteration settings.

The current Newton implementation also has these restrictions:

* Floating-base systems require an explicit free joint connecting the root body
  to the world.
* Joint velocity-limit constraints require ``pgs_mode="matrix_free"``.
* Friction modes other than ``"current"`` require
  ``pgs_mode="matrix_free"``.
* Joint limits are incompatible with the ``"tiled_contact"`` and
  ``"streaming"`` dense PGS kernels.
