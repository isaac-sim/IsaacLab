* Added a Franka Reach ``hold`` preset with end-effector pose-error feedback, a fixed episode goal,
  continued pose-success rewards, task-level Newton and Isaac Sim PhysX control profiles for
  quiet holding, and a separate RSL-RL experiment directory for its 38-input policies.
  Holding requires the Torch frontend and a trained or local holding checkpoint until pretrained
  policies are published. The environment browser filters domain presets by Warp compatibility
  and only offers pretrained checkpoints for declared published presets.
