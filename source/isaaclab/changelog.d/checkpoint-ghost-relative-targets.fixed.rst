* Fixed Checkpoint Ghosts copying a navigation task's relative targets from the reference. Each ghost now recomputes
  ``pos_command_b`` and ``heading_command_b`` from its own pose.
* Fixed Checkpoint Ghosts receiving the reference's turn command on velocity tasks that steer toward a heading. A ghost now
  copies the reference's sampled velocity and heading target and derives its own yaw rate from its own heading.
* Fixed Checkpoint Ghosts holding joints the actions do not drive, such as the fingers of a reaching arm, at different
  targets than the reference after a reset. A ghost now takes the reference's joint targets, so an identical
  checkpoint follows the reference.
* Fixed Checkpoint Ghosts running with different physics than the reference when a reset event randomizes mass,
  friction, or actuator gains per environment. Reset events named ``randomize_*`` are now disabled, like startup and
  interval events, and pose and velocity resets are still mirrored.
