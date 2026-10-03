* Added the state-based ``Isaac-Lift-SO101`` manager-based environment for lifting a
  tabletop cube to a commanded position under full gravity. Reused the Franka and Kuka
  lift command generator and contact-gated rewards with SO-101 scene bindings. Used
  fresh tabletop resets, compact state observations, arm targets within joint limits, and
  12-second episodes with resampled goals. Used a lightweight PPO configuration
  without a curriculum or camera observations.
* Added a shared lift MDP observation for end-effector-to-object displacement with
  configurable scene references.
