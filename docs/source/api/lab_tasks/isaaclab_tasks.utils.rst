isaaclab_tasks.utils
====================

.. automodule:: isaaclab_tasks.utils
   :members:
   :imported-members:

Reset-state sampling
--------------------

:class:`~isaaclab_tasks.utils.ResetStateSampler` shares success-guided sampling between
tasks without prescribing their geometry or state representation. A task creates a fixed
catalog of valid starts, and the sampler assigns one catalog row to each environment.
The configured :class:`~isaaclab_tasks.utils.SuccessMonitor` records recent episode outcomes
for each row. The default Beta and uniform mixture favors starts with intermediate success
while keeping every row eligible.

Generate and validate the catalog once when the environment starts. Procedural tasks can
store joint poses, object poses, and command metadata in device tensors indexed by row.
For cached simulation states, use
:meth:`~isaaclab.scene.InteractiveScene.get_state` with ``is_relative=True`` and restore
selected rows through :meth:`~isaaclab.scene.InteractiveScene.reset_to` with the same flag.
This preserves environment-relative positions and the public scene state format; no custom
asset serializer or private scene registries are needed. Catalog generation, acceptance
checks, and optional persistence remain task responsibilities.

Construct the sampler once, then select rows before resetting the physical scene:

.. code-block:: python

   from isaaclab_tasks.utils import ResetStateSampler, ResetStateSamplerCfg

   reset_sampler = ResetStateSampler(ResetStateSamplerCfg(), num_envs, row_count, device)

   # In the task's reset event, before clearing its previous success flags:
   rows = reset_sampler.reset(
       env_ids,
       success=episode_success[env_ids],
       valid=episode_length[env_ids] > 0,
   )
   # Restore physical state and command metadata indexed by rows using public asset APIs.

``success`` and ``valid`` contain one value per selected environment. Omit ``success`` for
the initial draw, or use ``valid`` to exclude zero-step manual resets. An optional ``row_mask``
restricts the next draw without changing which completed episodes are credited. Outcomes
always belong to the previous assignments, and a partial reset preserves other environments'
assignments. Task code must also clear commands, action targets, and attachment state that
are not represented in its physical state bank.

Keep evaluation starts fixed or uniformly sampled, and report success from complete task
starts separately from the adaptive training mixture. Sampling history belongs to the live
environment; ordinary policy checkpoints do not save it.

Sampling strategies
-------------------

.. automodule:: isaaclab_tasks.utils.sampling
   :members:
   :imported-members:
