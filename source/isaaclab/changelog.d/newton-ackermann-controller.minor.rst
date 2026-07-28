Added
^^^^^

* Added :class:`~isaaclab.controllers.AckermannController` and
  :class:`~isaaclab.envs.mdp.actions.AckermannAction` for controlling physical Ackermann-steered vehicles.

Changed
^^^^^^^

* Changed :class:`~isaaclab.controllers.DifferentialIKController` to delegate inverse-kinematics solves to Newton
  controllers while preserving its public configuration, command, output shape, and output dtype contracts. Solves
  now use float32 internal buffers; users that require float64 solver precision should retain the prior controller
  implementation until Newton provides a float64 path.
