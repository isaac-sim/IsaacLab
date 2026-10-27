* Changed BAM actuators to keep the joint's own viscous friction when
  :attr:`~isaaclab.actuators.BamMotorCfg.friction_viscous` is None.
* **Breaking:** Changed BAM actuators to raise at start-up when their USD actuators model gearbox backlash but
  the group sets no :attr:`~isaaclab.actuators.BamActuatorCfg.backlash_joint_template`. Set the template for
  such assets.
