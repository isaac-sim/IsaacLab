* Added :attr:`~isaaclab.sensors.camera.CameraCfg.modifiers` to process camera images with
  :class:`~isaaclab.utils.modifiers.ModifierCfg` chains keyed by camera output. The camera requests each
  chain input from its renderer, runs the chain once per captured image, resets it with the camera, and
  publishes the result in :attr:`~isaaclab.sensors.camera.CameraData.output`.
* Added :class:`~isaaclab.utils.modifiers.ModifierChain` to apply modifiers to one tensor outside the
  observation manager, constructing class modifiers from the shape of their first input.
* Added :meth:`~isaaclab.utils.modifiers.ModifierBase.close` so modifiers can release resources. Cameras
  close their modifiers when they are released or re-initialized.
* Added :attr:`~isaaclab.utils.modifiers.ModifierBase.output_dim` for modifiers that change the shape of their
  data, and :meth:`~isaaclab.utils.modifiers.ModifierBase.bind_sensor`, through which a camera passes itself to
  the modifiers it applies.
* Added :meth:`~isaaclab.managers.ObservationManager.close`, which closes class modifiers when the environment
  closes.
