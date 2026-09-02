Added
^^^^^

* Added a physics preset to the ``IsaacContrib-PickPlace-Locomanipulation-G1-Abs`` task so it can
  run on the Newton MJWarp backend via ``physics=newton_mjwarp``. The task previously assigned no
  ``sim.physics``, so the backend could not be selected at all. The ``default`` preset keeps the
  bare :class:`~isaaclab_physx.physics.PhysxCfg` the task used before, leaving PhysX unchanged.

  Bringing the task up on MJWarp needed three backend-specific adjustments:

  * The steering wheel's rim is a torus and MuJoCo has no concave mesh-mesh collision, so it is
    not graspable under MJWarp however it is approximated. The ``newton_mjwarp`` preset selects
    the graspable primitive from ``Isaac-Lift-Franka``, which runs MJWarp by default.
  * Newton matches contact sensors against the model's body labels, which keep the asset's
    intermediate grouping prim (``/Robot/left_hand/left_hand_index_0_link``). The task's pattern
    stopped at ``/Robot/`` and matched nothing, failing sensor initialization; the pattern is now
    preset per backend.
  * Newton resolves an omitted friction value to zero, so the robot needed an authored contact
    material to grip at all.
  * ``packing_table.usd`` authors its tabletop collider as a ``boundingCube``
    ``PhysicsCollisionAPI`` on an Xform rather than on mesh prims. Newton emits no shape for it,
    so the object fell through the table; an invisible static box reproduces the same bounding
    volume, offset for this task's lower table placement.

  Unlike the fixed-base humanoid tasks, gravity stays enabled and the solver follows the
  locomotion profile: this robot walks, so its lower-body policy needs real ground contact.

Fixed
^^^^^

* Fixed the locomanipulation lower-body action term running its frozen locomotion policy without
  ``torch.no_grad()``. Newton writes joint targets through Warp kernels, which reject a tensor
  that requires grad (``Can't get __cuda_array_interface__ on Variable that requires grad``), so
  the first ``env.step`` raised under MJWarp. PhysX writes through torch and was unaffected.

* Fixed the locomanipulation lower-body policy reading and writing joints in articulation order.
  The observation and action terms selected joints by regex, which resolves in the articulation's
  own order, and the backends do not agree on that order: PhysX enumerates the articulation
  breadth-first by tree depth (``left_hip_pitch, right_hip_pitch, waist_yaw, left_hip_roll, ...``)
  while Newton enumerates each limb chain depth-first (``left_hip_pitch, left_hip_roll,
  left_hip_yaw, left_knee, ...``). Under MJWarp the pretrained policy therefore received a permuted
  observation and its outputs were written to the wrong joints, and the robot fell over. Both terms
  now list their joints explicitly in the trained order and pass ``preserve_order=True``. The lists
  reproduce PhysX's regex resolution exactly, so the resolved joint indices — and PhysX behavior —
  are unchanged.
