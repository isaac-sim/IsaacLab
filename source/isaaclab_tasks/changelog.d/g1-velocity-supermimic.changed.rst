* Changed the motion tracking ``ArticulationKinematics`` to compute an optional selection of bodies, so only the joints
  between them and the floating base must be revolute.
* Changed the flat G1 SuperMimic agent to the settings of the official SuperMimic code: no world-model-only warm-up, no
  collection noise and BF16 matrix operations.
* Changed the velocity ``state_obs`` state to hold the world root position instead of the height above the environment
  origin, and dropped the foot velocities, which the differentiable task now takes from forward kinematics of the Newton
  model. Flat G1 SuperMimic checkpoints from before this change cannot be loaded.
* Changed the velocity SuperMimic actor to read the command in each observation frame, like the PPO policy, instead of
  through a separate goal input, whose encoder could saturate early in training and leave the policy blind to commands.
