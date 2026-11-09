* **Breaking:** Removed ``isaaclab_experimental.managers.SceneEntityCfg``. Use :class:`isaaclab.managers.SceneEntityCfg`,
  whose ``joint_ids_wp`` and ``body_ids_wp`` the Warp managers fill. Its ``joint_mask`` is removed: Warp terms loop
  over ``joint_ids_wp``.
* **Breaking:** Removed ``ManagerBasedEnvWarp.env_origins_wp``. Read ``scene.env_origins.warp``.
