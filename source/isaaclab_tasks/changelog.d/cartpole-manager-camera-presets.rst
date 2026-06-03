Changed
^^^^^^^

* Reworked the manager-based cartpole camera env into a preset-driven
  :class:`~isaaclab_tasks.core.cartpole.cartpole_manager_camera_env_cfg.CartpoleCameraEnvCfg`,
  mirroring the direct env. The camera data type and rendering backend (RTX,
  OmniverseRTX, Newton + Warp) are now selectable through the ``presets=``
  selector, e.g. ``presets=depth`` or ``presets=newton_renderer``.

Removed
^^^^^^^

* Removed the per-pipeline manager cartpole camera cfg subclasses
  ``CartpoleRGBCameraEnvCfg``, ``CartpoleDepthCameraEnvCfg``,
  ``CartpoleResNet18CameraEnvCfg`` and ``CartpoleTheiaTinyCameraEnvCfg`` along
  with their dedicated scene cfgs. Select the pipeline through the preset-based
  :obj:`Isaac-Cartpole-Camera` task instead, e.g. ``presets=rgb``,
  ``presets=depth``, ``presets=resnet18`` or ``presets=theia_tiny``.
