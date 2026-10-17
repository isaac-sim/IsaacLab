* Fixed :class:`~isaaclab_physx.sensors.ContactSensor` returning pre-reset contact points and, with Isaac Sim 6.1,
  pre-reset filtered friction forces and friction history when its data was read after
  :meth:`~isaaclab.scene.InteractiveScene.reset` and before the next physics step. Like the net forces, these
  buffers now keep their reset values until the sensor's next update.
