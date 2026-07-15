Changed
^^^^^^^

* Updated :obj:`~isaaclab_assets.robots.franka.FRANKA_PANDA_CFG` to use a menagerie-converted
  USD asset (``franka_panda.usda``) that carries identified Franka inertial parameters calibrated
  against Newton's actuator model, replacing the previous ``panda_instanceable.usd``.
* Merged the ``panda_shoulder`` and ``panda_forearm`` arm actuators in
  :obj:`~isaaclab_assets.robots.franka.FRANKA_PANDA_CFG` into a single ``panda_arm``
  :class:`~isaaclab.actuators.ImplicitActuatorCfg` with per-joint stiffness, damping, and
  armature values derived from the hardware datasheet and Drake's Franka model.
* Added a ``panda_finger2_passive`` actuator to
  :obj:`~isaaclab_assets.robots.franka.FRANKA_PANDA_CFG` (zero stiffness/damping) to model the
  passive finger joint driven through the hand's mimic coupling rather than an independent motor.
* Updated :obj:`~isaaclab_assets.robots.franka.FRANKA_PANDA_HIGH_PD_CFG` overrides to target
  the new ``panda_arm`` actuator key.
