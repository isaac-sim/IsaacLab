* Added the motion tracking ``state_obs`` observation preset and a differentiable motion tracking task for SuperMimic.
  Tracked-body poses and velocities come from differentiable forward kinematics of the simulated articulation, so the
  tracking rewards and terminations match the environment's managers; terminations use sigmoids around the
  environment's thresholds during imagination.
* Added the G1 motion tracking SuperMimic agent, available through ``rsl_rl_supermimic_cfg_entry_point``.
* Exposed the reference command to the actor and applied reference corrections in the simulator.
