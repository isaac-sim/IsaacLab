* Changed :meth:`~isaaclab.envs.ManagerBasedRLEnv.step` to reset on the ``reset_buf`` mask every step and to
  resample commands and run interval events on device masks, so a step no longer waits on the device to compact
  them into indices. Environments that override ``_reset_idx`` or record episodes still receive indices.
* Changed episode logs in ``extras["log"]`` to stay device tensors. A key keeps its last value on steps without a
  reset.
* **Breaking:** Built-in event, command, curriculum, action, observation and reward terms now take ``env_mask``, a
  :class:`~isaaclab.utils.warp.ProxyArray` boolean mask of shape ``(num_envs,)``, instead of ``env_ids``, and leave
  unselected environments unchanged. Code that calls them directly must pass a mask, for example from
  :func:`~isaaclab.utils.env_mask_from_ids`.
* **Breaking:** :meth:`~isaaclab.managers.CommandTerm._resample` now takes ``env_mask``. Managers and
  :meth:`~isaaclab.managers.ManagerTermBase.reset` take ``env_mask`` next to ``env_ids``. Terms whose function,
  ``reset`` or ``_resample_command`` declares ``env_mask`` run on every reset or update; terms that take
  ``env_ids`` keep working and are skipped when no environment is selected, at the cost of one device
  synchronization.
