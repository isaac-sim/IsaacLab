Added
^^^^^

* Added a physics preset to the ``IsaacContrib-PickPlace-GR1T2-Abs`` task so it can run on the
  Newton MJWarp backend via ``physics=newton_mjwarp``. Previously the task never assigned
  ``sim.physics``, so ``physics=newton_mjwarp`` failed with ``Unknown preset(s)``. The ``default``
  preset keeps the bare :class:`~isaaclab_physx.physics.PhysxCfg` the task used before, so PhysX
  behavior is unchanged. Everything below is scoped to the ``newton_mjwarp`` preset.

  Bringing the task up on MJWarp needed several backend-specific adjustments:

  * The steering wheel's collision meshes are authored as ``convexDecomposition`` and several
    decompose into degenerate slivers, so the MJWarp model failed to compile with ``mesh volume
    is too small``. Every collision mesh except the wheel rim and spokes is now approximated with
    a single convex hull; those two keep their decomposition so the wheel stays graspable.
  * ``GR1T2_HIGH_PD_CFG`` disables gravity per body, which Newton does not read. The equivalent
    ``mjc:gravcomp`` is requested instead, otherwise the unactuated legs and head are pulled down.
  * ``packing_table.usd`` authors its tabletop collider as a ``boundingCube`` ``PhysicsCollisionAPI``
    on an Xform rather than on mesh prims. Newton emits no shape for it, so objects fell through the
    table; an invisible static box reproduces the same bounding volume.
  * The hands are authored with no drive gains, deferring to the USD. Newton does not pick those up
    and a moving finger target sent the articulation to NaN on the first step, so the gains are
    authored explicitly. They cannot be copied from PhysX: MJWarp will not run the authored 17184
    stiffness under any solver configuration tried.
  * The legs and head are unactuated. PhysX never excites them because gravity is disabled, but
    under MJWarp they picked up energy and spun, so they get a posture drive.
  * The solver profile follows Newton's dexterous-hand example rather than the parallel-gripper
    tasks it was copied from. With the gripper values, closing a hand onto the wheel threw the
    object at 2.2 m/s; with eight substeps and 50 CCD iterations the same grasp moves it 2 mm.
