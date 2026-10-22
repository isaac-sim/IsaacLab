* **Breaking:** Removed ``ManagerCallSwitch``, ``ManagerCallMode`` and the ``MANAGER_CALL_CONFIG``
  environment variable. Warp environments construct their Warp managers directly.
* **Breaking:** Removed ``WarpGraphCache``. The Warp managers and environments record their stages through
  :class:`~isaaclab_experimental.utils.CapturedStage`, and the environments call the managers as the stable
  environments do.
* **Breaking:** Removed ``ManagerBasedRLEnvWarp.step_warp_termination_compute``. The environment calls
  :meth:`TerminationManager.compute <isaaclab_experimental.managers.TerminationManager.compute>` directly.
* **Breaking:** Removed ``DirectRLEnvWarp.step_warp_action``. :class:`~isaaclab_experimental.envs.DirectRLEnvWarp`
  calls the task methods in the order of the stable :class:`~isaaclab.envs.DirectRLEnv` and no longer records them.
  To keep recording them as CUDA graphs, decorate a task's ``_apply_action``, ``_get_dones``, ``_get_rewards``,
  ``_get_observations`` and ``_reset_idx`` with :func:`~isaaclab_experimental.utils.captured`.
* **Breaking:** Changed the Warp
  :meth:`ActionManager.process_action <isaaclab_experimental.managers.ActionManager.process_action>` to take a
  torch tensor, as the stable :meth:`~isaaclab.managers.ActionManager.process_action` does.
* Changed the Warp managers to decide CUDA graph capture per term: an operation records its capturable terms
  and runs each term decorated with ``@WarpCapturable(False)``, and each observation term with history, eagerly
  between the recorded parts, instead of running the whole operation eagerly. A class term's ``reset`` is
  recorded unless the ``reset`` method itself is decorated with ``@WarpCapturable(False)``.
* Changed the Warp manager-based environments to reset environments by boolean mask. The mask is converted
  to environment indices once per reset, and only when a command, curriculum or recorder term is active.
* Changed :class:`~isaaclab_experimental.envs.DirectRLEnvWarp` to skip its reset stage on steps where no
  environment terminated, deciding on the GPU (a conditional CUDA graph node, CUDA 12.4+), so a step no longer
  synchronizes with the host. With ``compute_final_obs``, ``extras["final_obs"]`` is now set on every step.
* Changed the Warp ``push_by_setting_velocity``, ``apply_external_force_torque``, ``reset_root_state_uniform``
  and ``randomize_rigid_body_com`` event terms to classes that allocate their buffers when the term is
  created. They read their ranges on every call, so a changed range applies to the next event.
* **Breaking:** Renamed :attr:`SceneEntityCfg.joint_mask <isaaclab_experimental.managers.SceneEntityCfg.joint_mask_wp>`
  to ``joint_mask_wp``, matching the ``_wp`` suffix of its other Warp fields, and added ``body_mask_wp``, the
  boolean mask of the selected bodies.
