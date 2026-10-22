* Added :class:`~isaaclab_experimental.utils.CapturedStage` and the :func:`~isaaclab_experimental.utils.captured`
  decorator, which run a Warp manager or environment stage eagerly or through CUDA graphs recorded from its own
  calls. A stage records on its first call without an eager warm-up, records again when an argument array or
  scalar changes or the physics backend rebinds its buffers, and runs eagerly on CPU or while
  :attr:`CapturedStage.enabled <isaaclab_experimental.utils.CapturedStage.enabled>` is False. The Warp
  environments enable capture.
* Added :func:`~isaaclab_experimental.utils.eager` for calls that cannot be recorded inside a captured stage.
  While the stage records, the call ends the current CUDA graph and runs eagerly before the next graph starts.
  Every replay repeats the call at the same place with the same arguments.
* Added the ``ISAACLAB_SYNC_DEBUG`` environment variable: ``ISAACLAB_SYNC_DEBUG=1`` raises on a hidden
  GPU-to-host synchronization inside an eagerly executed Warp environment stage.
* Added a boolean ``env_mask`` argument to :meth:`~isaaclab_experimental.envs.ManagerBasedEnvWarp.reset`.
* Added parameter predicates to :class:`~isaaclab_experimental.utils.warp.WarpCapturable`:
  ``@WarpCapturable(lambda params: ...)`` marks a term capturable only for the parameters the predicate
  accepts.
