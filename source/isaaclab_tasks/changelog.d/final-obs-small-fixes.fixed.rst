* Fixed the Humanoid AMP environment replacing the whole ``extras`` dict when computing observations, which dropped
  time-outs, episode logs and terminal observations set earlier in the step.
