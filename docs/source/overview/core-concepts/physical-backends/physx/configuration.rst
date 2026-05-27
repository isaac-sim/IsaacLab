PhysX Configuration
===================

PhysX scene-level settings live on :class:`~isaaclab_physx.physics.PhysxCfg`,
which replaces the Isaac Lab 2.x ``PhysxCfg`` imported from ``isaaclab.sim``.
In Isaac Lab 3.0, import ``PhysxCfg`` from ``isaaclab_physx.physics`` instead.
Per-actor settings (such as per-articulation iteration counts, contact
filtering, or material properties) continue to be authored on the USD schema;
see :doc:`../../schema_cfgs` for the schema-level configuration helpers.

The example below shows a representative ``PhysxCfg`` for a contact-rich
manipulation task:

.. code-block:: python

    from isaaclab.sim import SimulationCfg
    from isaaclab_physx.physics import PhysxCfg

    physx_cfg = PhysxCfg(
        solver_type=1,                       # TGS
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


Common Parameters
-----------------

The following list highlights the parameters that most often need tuning. The
full reference lives on :class:`~isaaclab_physx.physics.PhysxCfg`.

Solver Selection
^^^^^^^^^^^^^^^^

* ``solver_type``: ``1`` for **TGS** (Temporal Gauss-Seidel, default) or ``0``
  for **PGS** (Projective Gauss-Seidel). TGS is the recommended default for
  articulated robots; PGS can be more forgiving on stiff legacy assets.
* ``solve_articulation_contact_last``: alters the articulation solver order so
  that dynamic contact is resolved at the end of the solve. Useful for stiff
  gripping scenarios. For Isaac Lab 3.0, the supported Isaac Sim versions
  include this feature; it depends on PhysX schema support introduced in
  Isaac Sim 5.1.


Solver Iterations
^^^^^^^^^^^^^^^^^

* ``min_position_iteration_count`` / ``max_position_iteration_count``: clamp
  range applied to every actor's individual position-iteration count.
* ``min_velocity_iteration_count`` / ``max_velocity_iteration_count``: clamp
  range for velocity iterations.
* ``enable_external_forces_every_iteration``: applies external forces on every
  TGS position iteration. Reduces noisy velocity updates at additional
  compute cost; ignored with the PGS solver.


Contact and Stability
^^^^^^^^^^^^^^^^^^^^^

* ``enable_ccd``: continuous-collision detection for fast-moving bodies.
* ``enable_stabilization``: extra solver stabilization pass; recommended only
  for low-rate simulations, where the simulation timestep ``dt`` is larger
  than about ``1 / 30`` seconds (below about 30 Hz). This pass can make
  contact-sensor force magnitudes inaccurate, so keep it disabled if you rely
  on the contact sensor for force observations.
* ``bounce_threshold_velocity``: relative velocity threshold [m/s] above which
  contacts bounce.
* ``friction_offset_threshold``: contact point distance [m] at which friction
  forces are applied.
* ``friction_correlation_distance``: distance [m] used to merge nearby
  contacts into a single friction anchor.


GPU Buffer Sizing
^^^^^^^^^^^^^^^^^

PhysX on the GPU does **not** dynamically grow scene buffers, so undersized
buffers crash or silently drop contacts. The ``gpu_*`` fields on
:class:`~isaaclab_physx.physics.PhysxCfg` control these capacities. The
defaults are tuned for a few-hundred-environment locomotion task; large
multi-thousand-environment manipulation tasks usually need to raise:

* ``gpu_max_rigid_contact_count``
* ``gpu_max_rigid_patch_count``
* ``gpu_found_lost_pairs_capacity``
* ``gpu_found_lost_aggregate_pairs_capacity``
* ``gpu_total_aggregate_pairs_capacity``
* ``gpu_collision_stack_size``
* ``gpu_heap_capacity``

PhysX prints ``[PhysX]`` warnings when a buffer is exhausted; treat those as
hard failures and re-tune.


See Also
--------

* :class:`~isaaclab_physx.physics.PhysxCfg` — full parameter reference.
* :doc:`../../schema_cfgs` — schema-level configuration helpers.
* :doc:`../../multi_backend_architecture` — how PhysX plugs into the backend
  factory pattern.
* PhysX 5 SDK documentation:
  https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxSceneDesc.html
