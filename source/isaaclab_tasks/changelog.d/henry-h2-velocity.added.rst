* Added ``Isaac-Velocity-Flat-H2`` and ``Isaac-Velocity-Rough-H2`` with RSL-RL PPO configurations
  for training Unitree H2 velocity-tracking policies from scratch. Falls terminate on excessive
  root tilt, with an additional minimum root height on flat terrain.
* Added foot-flight, foot-separation, contact-conditioned knee-flexion, and walking yaw-momentum
  penalties for humanoid locomotion.
* The biped air-time reward optionally penalizes ongoing swings beyond its duration threshold.
