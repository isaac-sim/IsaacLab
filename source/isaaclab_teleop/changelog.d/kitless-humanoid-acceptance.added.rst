* Added task hooks to :class:`~isaaclab_teleop.IsaacTeleopCfg`: ``prepare_environment``,
  ``action_conditioner``, ``action_validator`` and ``require_initial_bimanual_hand_tracking``.
  The live and replay agents run them before creating simulation resources and before each step.
* Added the :attr:`~isaaclab_teleop.IsaacTeleopDevice.hand_tracking_presence` property, which
  reports left and right hand presence in the latest input frame.
