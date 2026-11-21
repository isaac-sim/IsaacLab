* Changed the SO101 IK-Abs teleoperation pipeline to use a full-pose clutch home. Its
  ``prepare_environment`` hook disables the reset joint noise for live teleoperation and HDF5
  recording; ordinary task construction keeps the joint-noise standard deviation of ``0.02``.
