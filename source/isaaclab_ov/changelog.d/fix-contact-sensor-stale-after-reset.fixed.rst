* Fixed :class:`~isaaclab_ov.sensors.ContactSensor` returning pre-reset contact forces, force histories and contact
  points when its data was read after :meth:`~isaaclab.scene.InteractiveScene.reset` and before the next physics
  step, as the observation that follows an environment reset does. These buffers now keep their reset values until
  the sensor's next update.
