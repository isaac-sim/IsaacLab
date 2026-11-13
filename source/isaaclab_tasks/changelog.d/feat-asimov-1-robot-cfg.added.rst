* Added the ``IsaacContrib-Velocity-Rough-Asimov1`` velocity-tracking task for the Asimov-1 humanoid,
  using the shared velocity configuration, reward terms, and observation delay. Robot-specific terms
  include episode-aware swing heights, foot-height rewards relative to the last ground contact,
  whole-body angular momentum, and joint-zero calibration offsets owned by the action term.
