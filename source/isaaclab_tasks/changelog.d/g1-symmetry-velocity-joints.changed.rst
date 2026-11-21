* Changed the flat G1 FlashSAC configuration to the tuned settings: a discount of 0.99, 64 updates per iteration,
  learning rates of 3e-4 with a cosine schedule, a short warm-up, an action range of 1 and left/right symmetry
  augmentation. It now reads the ``policy`` observation group, which the flat G1 environment provides.
* Moved the G1 symmetry augmentation settings into ``G1_SYMMETRY_CFG``, shared by the G1 PPO teacher and SAC agents.
