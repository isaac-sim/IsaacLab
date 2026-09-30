* Fixed :meth:`~isaaclab_experimental.envs.InteractiveSceneWarp.reset` with ``env_mask`` resetting deformable
  objects, surface grippers, and sensors in every environment, and skipping cable objects.
* Fixed the Warp :class:`~isaaclab_experimental.managers.ObservationManager` rejecting every observation group with
  ``Configuration for the term 'history_order' is not of type ObservationTermCfg``, which stopped all manager-based
  Warp environments from constructing. ``history_order`` is now read as a group setting, as in the stable manager.
