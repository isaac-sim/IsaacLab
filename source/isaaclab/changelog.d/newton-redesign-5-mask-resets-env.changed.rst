* Changed :meth:`~isaaclab.envs.ManagerBasedRLEnv.step` to reset on the ``reset_buf`` mask every step and to
  resample commands and run interval events on device masks, so a step no longer waited on the device to compact
  them into indices. Environments that override ``_reset_idx`` or record episodes still received indices.
* Changed resets to compute environment indices at most once, and only when a manager term or recorder took
  indices. Index-based terms shared them and were skipped when no environment was selected.
* Changed episode logs in ``extras["log"]`` to stay device tensors. A key kept its last value on steps without a
  reset.
* Changed :class:`~isaaclab.envs.mdp.randomize_joint_parameters` and
  :class:`~isaaclab.envs.mdp.randomize_fixed_tendon_parameters` to check when created that the configured limit
  ranges could not sample a lower limit above an upper limit, and to clamp each sampled lower limit to its upper
  limit on the device, instead of reading every sample on the host.
* Changed :func:`~isaaclab.utils.dict.class_to_dict` to keep Warp arrays as they are, like Torch tensors.
* **Breaking:** Built-in event, command, curriculum, action, observation and reward terms took ``env_mask``, a
  :class:`~isaaclab.utils.warp.ProxyArray` boolean mask of shape ``(num_envs,)``, instead of ``env_ids``, and left
  unselected environments unchanged. Code that calls them directly must pass a mask, for example from
  :func:`~isaaclab.utils.env_mask_from_ids`.
* **Breaking:** :meth:`~isaaclab.managers.ManagerTermBase.reset` took only ``env_mask``.
  :meth:`~isaaclab.managers.CommandTerm.reset` and :meth:`~isaaclab.managers.CommandTerm._resample` took
  ``env_mask``, plus the indices it selects when the caller resolved them, which an index-based
  ``_resample_command`` received. Terms whose function, ``reset`` or ``_resample_command`` declares ``env_mask`` ran
  on every reset or update; terms that take ``env_ids`` kept working and received the same full-reset selection as
  before.
* **Breaking:** :meth:`~isaaclab.managers.SceneEntityCfg.finalize` took the scene next to the device.
* **Breaking:** :meth:`~isaaclab.terrains.TerrainImporter.update_env_origins` took a Warp mask of the environments
  to update and moves for every environment, instead of indices and moves for the selected environments.
