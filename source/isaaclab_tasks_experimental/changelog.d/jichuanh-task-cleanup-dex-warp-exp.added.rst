* Added :class:`~isaaclab_tasks_experimental.core.reorient.config.shadow_hand.shadow_hand_warp_env.ShadowHandDirectWarpEnv`,
  so ``Isaac-Reorient-Cube-Shadow-Direct`` runs under ``--frontend warp``. Like the stable task, it
  commands the four finger tendons from the last four action columns.
* Added the ``Metrics/success_rate`` episode metric to the Warp in-hand reorientation environment,
  matching the stable Direct task.
