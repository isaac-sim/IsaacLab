* Added task action input activation to kitless ``isaaclab play``, with checkpoint-free
  input or Capture takeover during policy playback. Optional operator presentation used
  the independent OVXR render worker. Training and ``--no-teleop`` kept input inactive.
  RLinf rejected local input because its environments run in remote workers.
  Existing controller types, configuration paths, and task-specific APIs were preserved.
  Inactive input declarations did not prevent ordinary policy playback or training.
* Added ``--max_steps`` to policy playback for bounded interactive runs.
* Added ``--dataset_file``, ``--num_demos``, and ``--demo_export_mode`` to local playback for
  collecting HDF5 demonstrations through the same task configuration as policy playback,
  rejecting explicitly blank output paths before startup.
