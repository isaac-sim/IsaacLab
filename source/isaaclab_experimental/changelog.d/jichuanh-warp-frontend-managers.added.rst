* Added Warp twins of the ``UniformVelocityCommand``, ``UniformPoseCommand`` and ``NullCommand`` command
  terms. ``--frontend warp`` now runs command terms on the Warp command manager and records its compute and
  reset stages. The pose command logs ``Metrics/success_rate`` from its position and orientation thresholds,
  like the stable term.
* Added :meth:`~isaaclab_experimental.managers.CommandManager.get_command_wp` to read a command as a Warp
  array.
