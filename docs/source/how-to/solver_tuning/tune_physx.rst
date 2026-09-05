.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _physx-solver-tuning:

Tune the PhysX Solver
=====================

Use this guide after the PhysX backend can construct the task and its assets.
Before changing solver settings, reproduce one issue with a fixed initial
state, seed, and action sequence. Record the failure, contact behavior, and
task metric for that reproduction, then change one scene-level setting at a
time. The generated API documentation for
:class:`~isaaclab_physx.physics.PhysxCfg` remains authoritative for exact
fields and current defaults.

Start from an explicit baseline
-------------------------------

Make the scene-level PhysX choices visible in the task configuration first.
This baseline is a representative starting point for a contact-rich task, not a
universal recommendation:

.. code-block:: python

    from isaaclab.sim import SimulationCfg
    from isaaclab_physx.physics import PhysxCfg

    physx_cfg = PhysxCfg(
        solver_type=1,
        min_position_iteration_count=8,
        max_position_iteration_count=64,
        min_velocity_iteration_count=1,
        max_velocity_iteration_count=4,
        enable_ccd=False,
        enable_stabilization=False,
        bounce_threshold_velocity=0.2,
        friction_offset_threshold=0.04,
        friction_correlation_distance=0.025,
    )

    sim_cfg = SimulationCfg(dt=1 / 120, physics=physx_cfg)

Keep per-actor properties such as contact materials, articulation iteration
counts, and USD-authored collision settings on the schema configuration side;
scene-level PhysX tuning does not replace that authoring path.

Choose TGS or PGS
-----------------

Keep ``solver_type=1`` (TGS) as the first candidate for articulated robots and
contact-rich manipulation. TGS is the established default and usually gives the
best stability for modern Isaac Lab tasks. Compare ``solver_type=0`` (PGS) only
when the fixed reproduction still shows poor behavior after validating the
asset, controller, and contact geometry, or when a stiff legacy asset behaves
better with PGS.

If a grasping or stiff-contact workload remains sensitive after the solver
choice is fixed, test ``solve_articulation_contact_last`` on the same
reproduction and keep it only when it measurably improves the recorded contact
or task metric.

Tune solver iterations
----------------------

PhysX clamps each actor's solver work through the scene-level iteration ranges:
``min_position_iteration_count`` and ``max_position_iteration_count`` for
position iterations, plus ``min_velocity_iteration_count`` and
``max_velocity_iteration_count`` for velocity iterations.

Increase position iterations first when the fixed reproduction shows
penetration, joint constraint drift, or grasp instability. Increase velocity
iterations only when the same reproduction shows unstable restitution or a
velocity-level convergence issue that position iterations do not correct. Stop
when the measured behavior plateaus; more iterations cannot repair invalid
collision geometry, reset overlap, or a controller that injects instability.

Tune contacts and stability
---------------------------

Validate colliders, material properties, and reset overlap before changing
scene-wide contact settings. Then use the same fixed reproduction to tune:

* ``enable_ccd`` for fast-moving bodies that tunnel through thin geometry when
  using CPU dynamics. CCD is not supported with GPU dynamics; Isaac Lab forces
  it off and emits a warning if ``enable_ccd=True`` is requested on a GPU
  simulation.
* ``enable_stabilization`` only for low-rate simulations where ``dt`` is larger
  than about ``1 / 30`` seconds; this extra pass can make reported
  contact-sensor force magnitudes inaccurate, so keep it disabled when those
  forces are part of the observation or evaluation path.
* ``bounce_threshold_velocity`` [m/s] for the relative speed above which
  contacts bounce.
* ``friction_offset_threshold`` [m] for the contact distance at which PhysX
  starts applying friction forces.
* ``friction_correlation_distance`` [m] for the distance used to merge nearby
  contacts into one friction anchor.

Treat each of these as a measured contact-model change. Recheck penetration,
slip, bounce, contact counts, task success, and runtime after every adjustment.

Size GPU buffers
----------------

Most PhysX GPU scene-buffer capacities are fixed and do not grow dynamically.
When a scene exceeds one of these capacities, PhysX can drop contacts or fail
with a ``[PhysX]`` warning. Treat those warnings as hard failures for the
reproduction and increase only the capacity that overflowed.

The most common fixed capacities to raise for large vectorized or contact-rich
scenes are:

* ``gpu_max_rigid_contact_count``
* ``gpu_max_rigid_patch_count``
* ``gpu_found_lost_pairs_capacity``
* ``gpu_found_lost_aggregate_pairs_capacity``
* ``gpu_total_aggregate_pairs_capacity``
* ``gpu_collision_stack_size``

``gpu_heap_capacity`` is different: it sets the initial capacity of the GPU
and pinned-host-memory heaps, and PhysX allocates additional memory when those
heaps need to grow. Increase the initial heap capacity when measurements show
that repeated growth is undesirable, not because it is a fixed upper bound.

Measure the busiest reset and contact state, add task-specific headroom, and
then verify that the same fixed reproduction remains stable. Buffer sizing
cannot correct invalid geometry, solver divergence, or a controller issue.

See also
--------

* :class:`~isaaclab_physx.physics.PhysxCfg`
* :doc:`/source/overview/core-concepts/schema_cfgs`
* :ref:`physics-backends-physx`
* :doc:`/source/concepts/solver_differences`
