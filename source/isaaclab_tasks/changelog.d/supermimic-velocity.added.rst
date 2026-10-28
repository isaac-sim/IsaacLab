* Added the shared velocity ``state_obs`` observation preset and a robot-parameterized differentiable flat-ground
  velocity task for SuperMimic. Optional group-level presets add physical observations alongside the task policy
  observations. The task keeps privileged masses and contact labels outside actor inputs, integrates root rotations
  on SO(3), and advances contact clocks analytically. Torque, policy-rate acceleration, and imagined contact timing are documented
  approximations; per-term G1 tests compare the physical rewards and terminations with the actual managers.
