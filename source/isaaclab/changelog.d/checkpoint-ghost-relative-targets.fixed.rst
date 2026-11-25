* Fixed Checkpoint Ghosts copying a navigation task's relative targets from the reference. Each ghost now recomputes
  ``pos_command_b`` and ``heading_command_b`` from its own pose.
* Fixed Checkpoint Ghosts receiving the reference's turn command on velocity tasks that steer toward a heading. A ghost now
  copies the reference's sampled velocity and heading target and derives its own yaw rate from its own heading.
* Fixed Checkpoint Ghosts holding joints the actions do not drive, such as the fingers of a reaching arm, at different
  targets than the reference after a reset. A ghost now takes the reference's joint targets, so an identical
  checkpoint follows the reference.
* Fixed Checkpoint Ghosts running with different reset randomization than the reference, such as per-environment mass,
  friction, actuator gains, or a task's own sampled target. Each ghost slice now resets from the same random state as
  the reference, so it draws the reference's values.
* Fixed the reference's episode log being overwritten by the ghost resets, and ``--checkpoint_ghosts`` failing on a task
  with an observation group set to ``None``.
