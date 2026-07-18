Newton Manager Abstraction
==========================

Newton exposes multiple solver families, and Isaac Lab keeps that flexibility by
making each solver an implementation detail of a small
:class:`~isaaclab_newton.physics.NewtonManager` subclass. The simulation context
still sees a normal physics manager; the solver configuration decides which
manager class is used.

For most new Newton solvers, the integration surface is intentionally small:

* define a solver config that inherits from
  :class:`~isaaclab_newton.physics.NewtonSolverCfg`;
* point the config's ``class_type`` at a manager subclass;
* implement ``_build_solver()`` in that manager;
* set the three base-manager slots: ``_solver``, ``_use_single_state``, and
  ``_needs_collision_pipeline``.

The existing MuJoCo Warp, XPBD, Featherstone, and Kamino managers are examples
of this pattern.


Adding a Solver Manager
-----------------------

The solver config carries both user-tunable solver parameters and the manager
dispatch target:

.. code-block:: python

    from isaaclab_newton.physics import NewtonManager, NewtonSolverCfg
    from isaaclab.utils.configclass import configclass


    @configclass
    class MySolverCfg(NewtonSolverCfg):
        class_type: type[NewtonManager] | str = "{DIR}.my_solver_manager:NewtonMySolverManager"
        solver_type: str = "my_solver"
        iterations: int = 16


``NewtonCfg`` copies ``solver_cfg.class_type`` into its own ``class_type`` in
``__post_init__``. User code keeps the normal shape:

.. code-block:: python

    from isaaclab.sim import SimulationCfg
    from isaaclab_newton.physics import NewtonCfg

    sim_cfg = SimulationCfg(
        physics=NewtonCfg(
            solver_cfg=MySolverCfg(iterations=32),
            num_substeps=2,
        )
    )


The manager then owns solver construction:

.. code-block:: python

    from newton import Model
    from newton.solvers import SolverMySolver

    from isaaclab_newton.physics import NewtonManager


    class NewtonMySolverManager(NewtonManager):
        @classmethod
        def _build_solver(cls, model: Model, solver_cfg: MySolverCfg) -> None:
            NewtonManager._solver = SolverMySolver(model, iterations=solver_cfg.iterations)
            NewtonManager._use_single_state = False
            NewtonManager._needs_collision_pipeline = True


``_use_single_state`` tells the base manager whether the solver advances in
place or swaps input/output states. ``_needs_collision_pipeline`` tells the base
manager whether to allocate and pass Newton collision-pipeline contacts to the
solver. A solver with its own internal contact detector can set it to ``False``.

Optional Overrides
------------------

Most managers only implement ``_build_solver()``. Override more only when the
solver actually needs it:

* ``_initialize_contacts()``: allocate custom contact buffers or support an
  internal contact detector.
* ``_step_solver(state_0, state_1, control, substep_dt)``: change one substep of
  solver execution while keeping the base simulation loop.
* ``_simulate_physics_only()``: add per-step work around the base substep loop,
  such as rebuilding a BVH.
* ``step()``: handle solver-specific reset masks, graph capture, or model-change
  notification before delegating to the base manager.
* ``start_simulation()`` or ``instantiate_builder_from_stage()``: customize model
  building or post-finalize setup.
* ``_register_builder_attributes(builder)``: register solver-specific Newton
  custom attributes (particle, shape, body) on the builder before particles or
  finalize run. The active manager class invokes this hook from
  ``create_builder()``, ``start_simulation()``, and
  ``instantiate_builder_from_stage()``.
  :class:`~isaaclab_newton.physics.NewtonMPMManager` is the in-tree example —
  it registers ``mpm:young_modulus`` and the rest of the implicit MPM
  particle attributes.
* ``_prepare_builder_for_finalize(builder)``: normalize imported or replicated
  builder data right before ``ModelBuilder.finalize()``.
  :class:`~isaaclab_newton.physics.NewtonMPMManager` uses this to clear mass and
  inertia on kinematic bodies so implicit MPM treats them as massless colliders.
* ``_supports_cuda_graph_capture()``: return ``False`` to opt the solver out of
  CUDA graph capture and fall back to eager execution. Defaults to ``True``;
  :class:`~isaaclab_newton.physics.NewtonMPMManager` returns ``True`` only for a
  fixed grid, since sparse/dense MPM grids reallocate as particles move.
* ``_solver_specific_clear()``: release any class-level state owned by the
  solver manager.

Keep the manager name prefixed with ``Newton`` and the solver config grouped
with the other Newton solver configs so autocomplete and backend discovery stay
predictable.


Custom Coupled Solvers
----------------------

Coupled solvers use the same abstraction. Instead of wrapping one Newton solver,
a coupled manager constructs two or more sub-solvers and overrides
``_step_solver()`` to define the substep order.
That means a custom coupling usually needs only a config that stores existing
solver configs plus a manager that defines how data flows between them; the
component solvers can stay unchanged.

The MJWarp + VBD deformable manager is a concrete example:

* :class:`~isaaclab_contrib.deformable.CoupledMJWarpVBDSolverCfg` stores a
  ``rigid_solver_cfg`` for :class:`~isaaclab_newton.physics.MJWarpSolverCfg`, a
  ``soft_solver_cfg`` for :class:`~isaaclab_contrib.deformable.VBDSolverCfg`,
  and a ``coupling_mode``.
* ``NewtonCoupledMJWarpVBDManager._build_solver()`` constructs
  ``SolverMuJoCo`` and ``SolverVBD`` from those sub-configs.
* ``_step_solver()`` dispatches to either one-way or two-way coupling.
* The base ``NewtonManager`` still owns state allocation, substep iteration,
  Fabric synchronization, and reset/clear lifecycle.

The two-way MJWarp + VBD substep stays compact because it is expressed as a
short coupling algorithm:

.. admonition:: Algorithm: Two-Way MJWarp + VBD Substep
   :class: note

   **Inputs:** rigid body state, deformable particle state, and the shared
   Newton collision pipeline.

   **Output:** updated rigid body and deformable particle state for one Newton
   substep.

   1. **Reset force accumulators.**
      Clear the rigid body and particle force buffers before evaluating the
      next contact pass.

   2. **Detect coupled contacts.**
      Run Newton collision detection once over the current rigid and
      deformable state.

   3. **Apply soft-to-rigid reactions.**
      Inject body-particle contact reactions into ``body_f`` so the rigid
      bodies can be pushed back by the deformable contact penalties.

   4. **Advance the rigid solver.**
      Step the MJWarp rigid solver with the coupled contact forces applied.

   5. **Preserve shared contacts for the soft solve.**
      Clear particle forces written during the rigid step while keeping the
      detected contact information available.

   6. **Advance the deformable solver.**
      Step the VBD soft solver against the same coupled contacts.


This keeps the custom part focused on the coupling policy. The manager does not
need to reimplement scene loading, asset buffers, reset handling, or the outer
simulation loop.

.. figure:: ../../../../_static/newton/franka-mjwarp-vbd-coupling.png
   :align: center
   :figwidth: 480px
   :class: square-crop-figure
   :alt: Franka manipulating a deformable object with MJWarp and VBD coupling

   Franka manipulation using MJWarp for rigid bodies and VBD for the deformable
   object.

You can exercise this coupling path with the Franka soft-body lifting task:

.. code-block:: bash

   ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Lift-Soft-Franka --num_envs 1 --visualizer kit

For the surface-deformable cloth variant, use ``--task Isaac-Lift-Cloth-Franka``.


This environment configures
:class:`~isaaclab_contrib.deformable.CoupledMJWarpVBDSolverCfg` with
``coupling_mode="two_way"``.

Tuning the Franka Soft-Body Lift
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tune the coupled contact behavior before training a policy:

* Start with ``coupling_mode="two_way"``. Compared with one-way coupling, two-way
  coupling can prevent clipping more easily because body-particle contact
  penalties can push the robot back instead of only moving the deformable.
* Use a small scripted grasp/lift check before training to confirm that grasping
  is possible and to inspect what clips when the grasp fails.
* Lower the arm actuator stiffness enough that the arm can respond to contact
  penalties. Prefer the arm being pushed back over the gripper clipping into the
  deformable.
* Tune :attr:`~isaaclab_contrib.deformable.NewtonModelCfg.soft_contact_ke`
  first. Increase it only as much as needed to prevent clipping, then adjust
  :attr:`~isaaclab_contrib.deformable.NewtonModelCfg.soft_contact_mu` so the
  gripper can carry the object without requiring an obviously unphysical
  friction value. Use
  :attr:`~isaaclab_contrib.deformable.NewtonModelCfg.soft_contact_kd` for
  stabilization if contacts chatter.
* Tune the ``soft_contact_*`` values together with ``shape_material_*`` values
  because rigid shape material parameters also affect the effective contact.
* If ``soft_contact_ke`` is not sufficient, or ``soft_contact_mu`` must be
  unphysically high, tune the Franka arm and hand actuator stiffness and maximum
  effort. For the gripper command, fully close the fingers and let the actuator
  maximum effort limit the actual squeeze.
* If the deformable no longer visibly deforms, ``soft_contact_ke`` is likely too
  high.
* If contacts are unstable or missed, increase the deformable mesh resolution or
  increase ``particle_radius`` in the deformable material so contact is detected
  earlier from a larger distance.
* If the rigid shapes still clip through the deformable, increase
  :attr:`~isaaclab_contrib.deformable.VBDSolverCfg.iterations`; more VBD
  iterations can improve contact convergence.


When to Add a Coupled Manager
-----------------------------

Add a coupled manager when one solver cannot own the whole model step by itself:

* rigid bodies should use one solver while particles or cloth use another;
* contact detection is shared, but each solver consumes the contacts
  differently;
* you need a custom force, impulse, or state exchange between solvers;
* the substep order is part of the algorithm.

Use a normal single-solver manager when all physics can be advanced by one
Newton solver. Use a coupled manager only for the small amount of glue that is
truly solver-specific.
