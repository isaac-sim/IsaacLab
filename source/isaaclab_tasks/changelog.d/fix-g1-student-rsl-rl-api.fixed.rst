* Fixed the G1 depth-student distillation algorithm to follow the RSL-RL API pinned for SAC support: ``process_env_step``
  now returns the per-step logging values and ``update`` returns ``(loss_dict, stats_dict)``, so
  ``Isaac-Velocity-Rough-G1-Depth-Student`` trains instead of crashing in the runner.
