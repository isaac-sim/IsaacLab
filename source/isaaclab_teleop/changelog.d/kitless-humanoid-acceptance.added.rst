* Added task hooks to :class:`~isaaclab_teleop.IsaacTeleopCfg`: ``require_initial_bimanual_hand_tracking``
  and the experimental ``action_conditioner`` and ``action_validator``. The live, replay and recording
  agents apply them before each step.
* Added the :attr:`~isaaclab_teleop.IsaacTeleopDevice.hand_tracking_presence` property, which
  reports left and right hand presence in the latest input frame.
* Added ``--require_hand_replay_data`` to the MCAP replay agent. It counts the capture's recorded hand
  frames, retargets synchronously, and ends a run after that many frames without stepping the held last
  action. The run is a success when the applied actions changed and a failure otherwise; its report
  carries ``recorded_hand_frames``. A capture without a two-hand frame is rejected.
