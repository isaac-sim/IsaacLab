* Fixed shoelace grasp scoring for G1 and Dual-Franka to require inward finger contacts with the
  cable between the fingers on the same or adjacent tail segments, rejecting support from below
  while preserving the existing policy contact observations.
* Removed sustained shoelace hold credit after the two-second quality-weighted budget was exhausted,
  so remaining stationary did not keep earning retention reward.
