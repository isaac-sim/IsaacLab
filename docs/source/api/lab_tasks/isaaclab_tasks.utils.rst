isaaclab_tasks.utils
====================

.. automodule:: isaaclab_tasks.utils
   :members:
   :imported-members:

Reset-state sampling
--------------------

:class:`~isaaclab_tasks.utils.ResetStateSampler` assigns reset rows and tracks their
outcomes through :class:`~isaaclab_tasks.utils.SuccessMonitor`. Tasks own the physical
state catalog and restoration. The default Beta/uniform mixture favors intermediate
success while retaining coverage.

.. code-block:: python

   from isaaclab_tasks.utils import ResetStateSampler, ResetStateSamplerCfg

   sampler = ResetStateSampler(ResetStateSamplerCfg(), num_envs, row_count, device)
   # Before clearing old success flags or restoring selected environments:
   rows = sampler.reset(env_ids, success=episode_success[env_ids], valid=episode_length[env_ids] > 0)
   # Restore physical state, commands, controller targets and attachments for these rows.

Outcomes belong to the previous row assignments. Omit ``success`` for the initial draw;
``valid`` excludes zero-step/manual resets, and ``row_mask`` restricts the next draw.
Partial resets preserve other environments' assignments. Sampling history is local to the
live environment and is not included in ordinary policy checkpoints. Evaluate complete-task
starts separately from the adaptive training mixture.

Sampling strategies
-------------------

.. automodule:: isaaclab_tasks.utils.sampling
   :members:
   :imported-members:
