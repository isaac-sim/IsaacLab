* Added task-configured Capture teleoperation to kitless ``isaaclab play``. Tasks with
  ``teleop`` configured can run without a checkpoint, or override selected actions during policy playback.
  ``--no-teleop`` disables input, and training leaves teleoperation inactive. RLinf rejects
  local teleoperation because its environments run in remote workers.
* Added ``--max_steps`` to policy playback for bounded interactive runs.
