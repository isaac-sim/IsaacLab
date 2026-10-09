* **Breaking:** Aligned Franka Reach with the shared reach success reward and reset behavior and increased
  RSL-RL PPO exploration to match UR10. Existing Franka Reach policies should be retrained for
  the updated task; continuous tracking evaluations can disable the success reward and termination.
  The default reward configuration now uses the shared ``RewardsCfg`` and no longer includes
  ``end_effector_position_tracking_fine_grained``; add that term explicitly for custom tracking tasks.
* Tuned Franka Reach arm damping per joint and used bounded initial joint offsets supported by
  both frontends to avoid concentrating reset samples at joint limits, while retaining the shared robot
  asset and finger drives.
