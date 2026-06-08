Solver Comparison
=================

This page summarizes the user-visible behavioural differences between the
solvers shipped with the Isaac Lab physical backends. It's intended as a
porting reference: most tasks reuse the same USD asset across backends, but
contact, friction, and stabilization behave differently enough that retuning
is usually required when moving from one solver to another.

The solvers covered are:

* **PhysX TGS** — default solver for the :doc:`PhysX backend <physx/index>`
  (Temporal Gauss-Seidel). PhysX also exposes a Projective Gauss-Seidel
  (PGS) variant via :attr:`~isaaclab_physx.physics.PhysxCfg.solver_type`,
  which behaves similarly for the purposes of this comparison.
* **Newton MuJoCo-Warp (MJWarp)** — primary :doc:`Newton solver <newton/mjwarp-solver>`,
  configured by :class:`~isaaclab_newton.physics.MJWarpSolverCfg`.
* **Newton Kamino** — beta P-ADMM :doc:`Newton solver <newton/kamino-solver>`,
  configured by :class:`~isaaclab_newton.physics.KaminoSolverCfg`.

Newton additionally ships ``FeatherstoneSolverCfg`` and ``XPBDSolverCfg``;
neither is wired into an Isaac Lab task at the time of writing and they
are omitted from this comparison.


Friction Model
--------------

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Solver
      - Friction handling
    * - PhysX TGS
      - Coulomb friction with **patch-based** anchors. Tangential forces are
        merged across nearby contacts via
        :attr:`~isaaclab_physx.physics.PhysxCfg.friction_correlation_distance`
        and applied above
        :attr:`~isaaclab_physx.physics.PhysxCfg.friction_offset_threshold`.
        The friction cone is always pyramidal.
    * - MJWarp
      - MuJoCo's friction model. The friction cone shape is selectable via
        :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.cone`
        (``"pyramidal"`` or ``"elliptic"``). The tangential-to-normal
        impedance ratio is exposed as
        :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.impratio`.
    * - Kamino
      - Per-contact friction resolved inside the P-ADMM solve. Contact
        warm-starting is selectable via
        :attr:`~isaaclab_newton.physics.KaminoSolverCfg.padmm_contact_warmstart_method`;
        the validated presets use ``"geom_pair_net_force"``.

**Porting implication.** Tasks tuned for PhysX's patch friction can feel
"grippier" than MJWarp's per-contact friction at the same friction
coefficient. When moving a manipulation task from PhysX to MJWarp, expect to
raise friction coefficients and consider switching ``cone`` to ``"elliptic"``
for stiffer contact stacks.


Contact Detection and Resolution
--------------------------------

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Solver
      - Collision pipeline
    * - PhysX TGS
      - PhysX's built-in broadphase + narrowphase, with optional continuous
        collision detection via
        :attr:`~isaaclab_physx.physics.PhysxCfg.enable_ccd`. Pre-sized GPU
        buffers (``gpu_max_rigid_contact_count`` etc.) cap the number of
        contacts per step; oversubscription is a hard error.
    * - MJWarp
      - Two modes selected by
        :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.use_mujoco_contacts`:
        MuJoCo's internal contact pipeline (default) or Newton's
        :class:`~isaaclab_newton.physics.NewtonCollisionPipelineCfg`. The two
        are mutually exclusive. GJK/EPA iteration count is exposed via
        :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.ccd_iterations`.
    * - Kamino
      - Optionally uses Kamino's internal collision detector (``"primitive"``
        or ``"unified"``) via
        :attr:`~isaaclab_newton.physics.KaminoSolverCfg.use_collision_detector`,
        otherwise falls back to Newton's :class:`CollisionPipeline`.
        Contact penetration margin is set by
        :attr:`~isaaclab_newton.physics.KaminoSolverCfg.constraints_delta`.

**Porting implication.** A task that runs with ``--enable_ccd`` on PhysX
won't get the same protection on MJWarp/Kamino at large ``dt`` — Newton's
CCD is convex GJK/EPA, not the swept-shape CCD PhysX uses. The mitigation
on Newton is shorter ``dt`` or higher
:attr:`~isaaclab_newton.physics.NewtonCfg.num_substeps`.


Restitution and Bounce
----------------------

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Solver
      - Restitution
    * - PhysX TGS
      - Restitution coefficient is per-material on the USD shape. A global
        velocity threshold
        :attr:`~isaaclab_physx.physics.PhysxCfg.bounce_threshold_velocity`
        suppresses restitution below ~0.5 m/s by default.
    * - MJWarp
      - Restitution follows the MJCF model translated from USD. There is no
        bounce-threshold gate — small-velocity contacts can still bounce
        unless you reduce the per-material restitution.
    * - Kamino
      - Restitution is contained in the contact constraint set; behaviour is
        similar to MJWarp.

**Porting implication.** Tasks that rely on PhysX's bounce-threshold to
suppress jitter (e.g. footstep contact on flat ground) may show contact
chatter on Newton until restitution coefficients are reduced.


Constraint Stabilization
------------------------

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Solver
      - Stabilization
    * - PhysX TGS
      - Implicit, via the TGS solver. An optional extra pass is enabled by
        :attr:`~isaaclab_physx.physics.PhysxCfg.enable_stabilization` (note:
        corrupts contact-sensor force magnitudes).
    * - MJWarp
      - Implicit, set by the MuJoCo solver's ``solref``/``solimp`` per
        constraint. No top-level toggle.
    * - Kamino
      - Explicit Baumgarte stabilization with separate gains for joint
        bilaterals (:attr:`~isaaclab_newton.physics.KaminoSolverCfg.constraints_alpha`),
        joint-limit unilaterals
        (:attr:`~isaaclab_newton.physics.KaminoSolverCfg.constraints_beta`),
        and contact unilaterals
        (:attr:`~isaaclab_newton.physics.KaminoSolverCfg.constraints_gamma`).


Solver Convergence
------------------

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Solver
      - Iteration model
    * - PhysX TGS
      - Two iteration counts, per actor: position
        (``min/max_position_iteration_count``) and velocity
        (``min/max_velocity_iteration_count``). The solver takes the largest
        actor's count clamped to the scene range. No global convergence
        tolerance.
    * - MJWarp
      - :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.iterations` outer
        Newton/CG iterations and
        :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.ls_iterations` line
        searches per outer iteration. Convergence gate:
        :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.tolerance` (default
        ``1e-6``).
        :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.ls_parallel` switches
        line search to parallel execution at a small accuracy cost.
    * - Kamino
      - P-ADMM with separate
        :attr:`~isaaclab_newton.physics.KaminoSolverCfg.padmm_primal_tolerance`,
        :attr:`~isaaclab_newton.physics.KaminoSolverCfg.padmm_dual_tolerance`,
        and
        :attr:`~isaaclab_newton.physics.KaminoSolverCfg.padmm_compl_tolerance`
        gates, capped at
        :attr:`~isaaclab_newton.physics.KaminoSolverCfg.padmm_max_iterations`.
        Acceleration and warm-starting are tunable.


Articulation Coordinates
------------------------

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Solver
      - Coordinate representation
    * - PhysX TGS
      - **Reduced-coordinate** articulations (joint-space, Featherstone-like).
        Joint state is the canonical truth; body transforms are derived via
        forward kinematics.
    * - MJWarp
      - Reduced-coordinate, computed by the MuJoCo solver.
    * - Kamino
      - **Maximal-coordinate**: each body has its own free-body state,
        constraints are enforced via Baumgarte stabilization. Resets go
        through a dedicated FK pass (:attr:`~isaaclab_newton.physics.KaminoSolverCfg.use_fk_solver`)
        so maximal body poses match the reduced joint state after a state
        write.

**Porting implication.** Kamino is more sensitive to inconsistent reset
state — joint positions and body poses must agree, which Isaac Lab's asset
write paths handle but custom reset code can break.


Substepping and Timestep
------------------------

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Solver
      - Substep model
    * - PhysX TGS
      - PhysX runs at the simulation ``dt``. No external substep counter;
        internal substepping is per-actor.
    * - MJWarp
      - Top-level :attr:`~isaaclab_newton.physics.NewtonCfg.num_substeps`
        controls how many solver substeps run per Isaac Lab step. Effective
        solver ``dt`` is ``SimulationCfg.dt / num_substeps``.
    * - Kamino
      - Same :attr:`~isaaclab_newton.physics.NewtonCfg.num_substeps` knob.
        Validated Kamino task presets typically use 1–2 substeps; expect to
        raise this for contact-heavy tasks.


GPU Buffers and Throughput
--------------------------

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Solver
      - Memory model
    * - PhysX TGS
      - Static GPU buffers sized at construction. Undersized buffers cause
        crashes or dropped contacts at scale. See the
        :doc:`PhysX configuration page <physx/configuration>` for the
        ``gpu_*`` knobs.
    * - MJWarp
      - Pre-allocated per-environment limits:
        :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.njmax`
        (constraint rows) and
        :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.nconmax` (contact
        points). The remainder of state lives in dynamically-sized Warp
        arrays.
    * - Kamino
      - Inherits MJWarp's pre-allocation pattern via Newton, plus its own
        contact-pair allocator
        (:attr:`~isaaclab_newton.physics.KaminoSolverCfg.collision_detector_max_contacts_per_pair`)
        when using the internal collision detector.


Porting Checklist
-----------------

When moving a task between solvers:

1. **Re-validate contact behavior.** Run the task at the smallest
   ``num_envs`` with a visualizer; watch for new penetration or jitter
   before scaling up.
2. **Retune friction.** PhysX patch friction and MJWarp per-contact friction
   are not interchangeable at the same coefficient.
3. **Retune restitution.** MJWarp/Kamino have no bounce-threshold gate;
   reduce per-material restitution to suppress jitter.
4. **Choose substeps.** PhysX → Newton typically needs at least 1–2
   substeps for contact-heavy tasks; manipulation tasks may need more.
5. **Watch for CCD differences.** Tasks that relied on PhysX swept CCD
   should either reduce ``dt`` or raise ``num_substeps`` on Newton.
6. **For Kamino specifically**, also validate reset behaviour and consider
   Baumgarte gains if you see drift on joint or contact constraints.
