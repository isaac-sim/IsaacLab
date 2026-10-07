* Added :attr:`~isaaclab.sensors.camera.CameraCfg.modifiers` to process camera images with
  :class:`~isaaclab.utils.modifiers.ModifierCfg` chains keyed by camera output. The camera requests each
  chain input from its renderer, runs the chain once per captured image, resets it with the camera, and
  publishes the result in :attr:`~isaaclab.sensors.camera.CameraData.output`.
* Added :class:`~isaaclab.utils.modifiers.ModifierChain` to apply modifiers to one tensor outside the
  observation manager, constructing class modifiers from the shape of their first input.
