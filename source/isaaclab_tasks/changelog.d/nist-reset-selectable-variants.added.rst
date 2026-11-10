* Added ``IsaacContrib-Factory-Variant-Franka`` with twenty reset-selectable assembly pairs,
  a shared validated reset-state bank, beta sampling, and point-cloud observations for a SimBa policy.
  The task used the public Newton/MuJoCo Warp backend without a custom Newton dependency.
  Explicit sampling seed, assembly order, and observation-only robot USD settings allowed compatible
  checkpoints to regenerate their input points without loading saved point clouds.
