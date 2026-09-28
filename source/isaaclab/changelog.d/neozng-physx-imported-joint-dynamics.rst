Fixed
^^^^^

* Fixed :class:`~isaaclab.sim.converters.UrdfConverter` and :class:`~isaaclab.sim.converters.MjcfConverter`
  dropping the joint friction and damping from the PhysX description of converted assets. The URDF importer did
  not convert them (isaac-sim/IsaacSim#841), and the MJCF importer converted ``frictionloss`` to the legacy,
  load-proportional ``physxJoint:jointFriction`` coefficient and dropped ``damping``. The ``physx`` physics variant
  and flat output now carried them per joint axis through ``PhysxJointAxisAPI``, together with the armature of
  MJCF ball and D6-folded joints, and the MJCF converter warned about passive joint springs, which PhysX cannot
  represent. Select ``physx`` through :attr:`~isaaclab.sim.converters.AssetConverterBaseCfg.physics_variant` to
  simulate them with PhysX, and delete assets converted earlier into a reused ``usd_dir`` so that they are
  converted again.
* URDF ``damping`` and ``friction`` acted on PhysX as passive joint friction in addition to the drive and actuator
  gains, as on Newton. To simulate a URDF asset without them, set ``friction``, ``dynamic_friction``, and
  ``viscous_friction`` of its actuator configuration to ``0.0``.
