.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _solver-differences:

Solver Differences
==================

Isaac Lab presents a common asset and task interface across its physics
backends, but that interface does not make the underlying solver models
interchangeable. PhysX and the Newton solvers can use the same USD asset while
producing different trajectories because they construct contacts, represent
state, stabilize constraints, and allocate work differently.

This comparison covers the PhysX solver configured by
:class:`~isaaclab_physx.physics.PhysxCfg`, Newton MuJoCo-Warp (MJWarp)
configured by :class:`~isaaclab_newton.physics.MJWarpSolverCfg`, and Newton
Kamino configured by :class:`~isaaclab_newton.physics.KaminoPADMMSolverCfg` or
:class:`~isaaclab_newton.physics.KaminoDVISolverCfg`. OvPhysX is a separate,
experimental backend with its own limitations; it is not another solver mode
of the PhysX backend discussed here.

Use this page to understand the model differences. For the controls and a
measured tuning procedure, use :doc:`/source/how-to/solver_tuning/tune_mjwarp` or
:doc:`/source/how-to/solver_tuning/tune_kamino`; the generated solver configuration APIs
are the reference for exact fields and defaults.

Why solver settings do not translate directly
----------------------------------------------

A configuration value has meaning only within the solver that consumes it.
For example, PhysX scene controls do not configure a Newton solver, and
MJWarp's MuJoCo-specific controls do not configure PhysX. The common Isaac Lab
configuration selects a backend and solver; it does not translate a numerical
setting into an equivalent physical effect.

Even settings with similar names can participate in different contact models
or convergence criteria. Porting therefore starts by preserving the mechanical
model, collision geometry, material bindings, reset state, actuator behavior,
and control period. Then compare the resulting physical behavior and tune the
target solver with its own controls.

Friction and contact
--------------------

PhysX uses patch-based Coulomb friction: nearby contacts can be correlated
into a friction patch. Its friction-correlation and offset controls are part
of :class:`~isaaclab_physx.physics.PhysxCfg`. MJWarp uses MuJoCo's contact
model, including its selectable pyramidal or elliptic friction cone and
frictional impedance ratio. Kamino resolves hard frictional contacts in its
maximal-coordinate constrained dynamics solve and exposes contact-material
mixing and warm-start configuration through its generated API.

The collision path also differs. PhysX exposes continuous collision detection
through :attr:`~isaaclab_physx.physics.PhysxCfg.enable_ccd`, but Isaac Lab
disables that option when GPU dynamics is enabled. MJWarp can use either its
internal MuJoCo contact path or Newton's collision pipeline; those modes are
mutually exclusive. Kamino can use Newton's collision pipeline or, when
enabled, its own internal collision detector. These choices determine which
capacity and collision controls are active.

**Porting implication.** Equal material coefficients do not guarantee equal
slip, grasp stability, or contact counts. Revalidate collision geometry and
contact behavior before tuning friction. A task that depended on PhysX CCD
needs an independently validated timestep and collision strategy on Newton;
MJWarp's ``ccd_iterations`` is a convex GJK/EPA convergence limit, not a
PhysX-style CCD switch.

Restitution and stabilization
-----------------------------

PhysX applies restitution using per-material properties and a scene-level
bounce threshold. Its optional stabilization pass can improve large-timestep
behavior, but can make reported contact-sensor forces inaccurate. MJWarp
expresses contact compliance through its MuJoCo model and solver formulation;
Kamino exposes Baumgarte stabilization separately for bilateral joints,
unilateral joint limits, and unilateral contacts.

These are different mechanisms, so copying a restitution or stabilization
setting does not preserve resting-contact behavior. The exact configuration
surface belongs in the :class:`~isaaclab_physx.physics.PhysxCfg`,
:class:`~isaaclab_newton.physics.MJWarpSolverCfg`, and
:class:`~isaaclab_newton.physics.KaminoConstraintsCfg` API references.

**Porting implication.** Validate bounce, penetration, chatter, and measured
contact forces in the target solver. Do not use a stabilization or restitution
control as a substitute for valid collision geometry, reset state, or material
properties.

Coordinates and state consistency
---------------------------------

PhysX articulations and MJWarp use reduced joint coordinates for their
articulated state. Kamino solves rigid multi-body systems in maximal
coordinates with constraints. Its reset path can reconcile body poses with
reduced joint state through forward kinematics; whether that path is used is
determined by :attr:`~isaaclab_newton.physics.KaminoPADMMSolverCfg.use_fk_solver`
and the articulation structure.

Isaac Lab asset APIs maintain the public state interface across the backends,
but custom reset code can still write a state that is inconsistent with the
target solver's representation.

**Porting implication.** Use the asset write APIs for resets and validate the
first step with a fixed state. For Kamino, treat disagreement between joint
state and body poses as a reset-modeling problem before tuning solver gains.

Timesteps and convergence
-------------------------

PhysX advances at the simulation timestep and uses actor iteration counts
within scene limits. MJWarp and Kamino each run
:attr:`~isaaclab_newton.physics.NewtonCfg.num_substeps` solver substeps per
physics tick, so their solver timestep is the simulation timestep divided by
the substep count. More substeps change integration work; they do not
translate a PhysX iteration setting into a Newton equivalent.

Substeps do not inherently refresh Newton-pipeline contacts. When that
pipeline is active, :attr:`~isaaclab_newton.physics.NewtonCfg.collision_decimation`
can re-collide within a physics tick only when it is positive and less than
``num_substeps``; the default value of zero collides once at the start of each
tick. This setting does not apply when MJWarp uses its internal MuJoCo contact
path or Kamino uses its internal collision detector.

MJWarp provides outer iterations, line-search iterations, and a residual
tolerance. Kamino's P-ADMM and DVI modes have different iteration and
convergence controls. PhysX has its own position and velocity iteration
model. Consult the generated APIs and the focused how-to pages for current
controls instead of copying another solver's defaults.

**Porting implication.** Compare a fixed reproduction at the intended policy
period, then separately test timestep, substeps, and solver convergence. An
iteration increase cannot correct invalid contact geometry, incompatible reset
state, or an unstable controller.

Capacity and memory
-------------------

PhysX allocates GPU buffers for scene limits, including rigid-contact capacity.
MJWarp has per-world contact and constraint limits such as ``nconmax`` and
``njmax``. Kamino has per-world contact allocation controls in addition to the
capacity used by its selected collision path. Their units and failure modes are
solver-specific, even when every environment contains the same asset.

**Porting implication.** Measure the target solver's busiest reset and contact
states before scaling environment count. Increase an observed overflowing
capacity first; capacity changes cannot repair an invalid contact model or
convergence problem.
