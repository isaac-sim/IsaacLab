* Added the ``ISAACLAB_WARP_CAPTURE`` environment variable: ``ISAACLAB_WARP_CAPTURE=0`` runs every Warp
  environment stage eagerly, for debugging and eager-versus-captured comparisons.
* Added the ``ISAACLAB_SYNC_DEBUG`` environment variable: ``ISAACLAB_SYNC_DEBUG=1`` raises on a hidden
  GPU-to-host synchronization inside an eagerly executed Warp environment stage.
* Added a boolean ``env_mask`` argument to :meth:`~isaaclab_experimental.envs.ManagerBasedEnvWarp.reset`.
* Added :meth:`~isaaclab_experimental.utils.WarpGraphCache.call_steps` and
  :meth:`ManagerBase.stage_steps <isaaclab_experimental.managers.ManagerBase.stage_steps>`, which split a
  manager stage into per-term steps that are recorded into CUDA graphs or run eagerly.
* Added parameter predicates to :class:`~isaaclab_experimental.utils.warp.WarpCapturable`:
  ``@WarpCapturable(lambda params: ...)`` marks a term capturable only for the parameters the predicate
  accepts. A class term can also set ``self._warp_capturable`` in ``__init__``. The Warp managers decide
  capture for each configured term instance.
