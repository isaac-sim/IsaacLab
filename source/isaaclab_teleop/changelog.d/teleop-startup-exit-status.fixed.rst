* Fixed ``isaaclab teleop run`` exiting with status 0 when the environment or the teleop device could not
  be created. The original exception now propagates after cleanup.
