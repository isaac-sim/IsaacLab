* Added Warp twins of the ``UniformVelocityCommand``, ``UniformPoseCommand`` and ``NullCommand`` command
  terms. ``--frontend warp`` now runs command terms on the Warp command manager and records its compute and
  reset stages. The pose command logs ``Metrics/success_rate`` from its position and orientation thresholds,
  like the stable term.
* Added :meth:`~isaaclab_experimental.managers.CommandManager.get_command_wp` to read a command as a Warp
  array.
* Added a Warp twin of the ``height_scan`` observation, so the Rough velocity tasks run with
  ``--frontend warp``. It reads a ray caster refreshed on the host, so it runs eagerly.
* Added support for Warp terms defined outside Isaac Lab on the Warp frontend: class terms that subclass
  :class:`~isaaclab_experimental.managers.ManagerTermBase` and function terms decorated with
  :class:`~isaaclab_experimental.utils.warp.WarpCapturable` are used as configured.
* Added ``set_term_cfg`` and ``get_term_cfg`` to the Warp
  :class:`~isaaclab_experimental.managers.TerminationManager` and
  :class:`~isaaclab_experimental.managers.CommandManager`. Setting a term configuration records the manager's
  stages again, so a curriculum that changes a termination parameter or a command range through
  ``modify_term_cfg`` takes effect under CUDA graph capture.
