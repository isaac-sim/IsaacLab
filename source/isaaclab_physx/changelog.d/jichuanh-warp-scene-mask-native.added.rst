* Added ``env_mask`` support to the Newton-native actuator reset of PhysX articulations, so
  :meth:`~isaaclab.actuators.ActuatorCollection.reset` with a mask resets the native actuator state of only the masked
  environments, without converting the mask to indices.
* Added ``env_mask`` to :meth:`~isaaclab_physx.assets.SurfaceGripper.reset`, which converts the mask to the
  indices the gripper view takes.
