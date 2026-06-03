Changed
^^^^^^^

* Consolidated the direct cartpole camera env and config into a single
  ``CartpoleCameraEnv`` / ``CartpoleCameraEnvCfg`` pair (in
  ``cartpole_direct_camera_env`` / ``cartpole_direct_camera_env_cfg``). The
  separate ``CartpoleCameraPresetsEnv`` env class and
  ``cartpole_direct_camera_presets_env`` /
  ``cartpole_direct_camera_presets_env_cfg`` modules were removed; frame
  stacking is now built into ``CartpoleCameraEnv`` and stays gated on the
  Newton + Warp backend combo. ``CartpoleCameraEnv`` now subclasses
  :class:`~isaaclab_tasks.core.cartpole.cartpole_direct_env.CartpoleEnv` and
  ``CartpoleCameraEnvCfg`` subclasses
  :class:`~isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg.CartpoleEnvCfg`,
  overriding only the camera-specific fields to remove duplication.

Removed
^^^^^^^

* Removed the orphaned per-datatype direct cartpole camera cfg subclasses
  ``CartpoleDepthCameraEnvCfg``, ``CartpoleAlbedoCameraEnvCfg``,
  ``CartpoleSimpleShadingConstantCameraEnvCfg``,
  ``CartpoleSimpleShadingDiffuseCameraEnvCfg`` and
  ``CartpoleSimpleShadingFullCameraEnvCfg``. These were left over from the
  pre-preset design and no longer back any task. Select the datatype through
  the preset-based :obj:`Isaac-Cartpole-Camera-Direct` task instead, e.g.
  ``presets=depth`` or ``presets=albedo``.
* Removed ``CartpoleRGBCameraEnvCfg`` from
  ``isaaclab_tasks.core.cartpole.cartpole_direct_camera_env_cfg``. The cartpole
  camera showcase now defines its own equivalent base cfg.
