Added
^^^^^

* Added the state-based ``Isaac-Lift-SO101`` manager-based environment for lifting and
  holding a tabletop cube with the SO-101 arm under full gravity. Used fresh tabletop
  resets, compact state observations, absolute joint targets, and a lightweight PPO
  configuration without a curriculum or camera observations.
* Added shared lift MDP terms for end-effector-to-object displacement and lift-and-hold
  rewards with configurable scene references, height and speed thresholds, and hold durations.
