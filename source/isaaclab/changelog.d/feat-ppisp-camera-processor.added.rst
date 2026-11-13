* Added ``CameraCfg.exposure`` and :class:`~isaaclab.sensors.ExposureCfg`. With ``mode="neutral"``, camera
  construction disables RTX automatic exposure and authors unit exposure on the camera prims before the scene is
  cloned, so color outputs keep the scene's linear radiance. ``"default"`` keeps the authored exposure.
* Added an optional ``prepare_camera`` class method to camera tensor processors, which camera construction calls
  before the scene is cloned or exported to resolve the processor configuration or remove the processor.
