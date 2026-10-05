* Added Newton MJWarp Franka and KUKA-Allegro cube-stacking tasks with validated
  physical reset banks, cube-role permutations, and target-rate sampling through
  the shared success monitor. Play mode used randomized table starts only.
* Added Franka RGB PPO and camera-distillation tasks with robot proprioception,
  standard RSL-RL CNNs, calibration and photometric randomization, and clipped
  teacher labels. Privileged object state stayed outside the deployed actor.
  The critic and teacher reused the state task's 100-input observation contract
  and model configuration; older 109-input teachers require retraining.
* Configured the Franka task with the standard flat asset's MuJoCo payload and
  passive mimic finger, calibrated impedance, and gravity compensation.
* Preserved the published state actors' observation order and used stable,
  released-stack success checks. Shared default ground visuals, colored cubes,
  and an invisible tabletop contact surface kept physics and rendering aligned.
