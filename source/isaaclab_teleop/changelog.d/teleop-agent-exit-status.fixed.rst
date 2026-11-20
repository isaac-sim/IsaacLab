* Fixed ``teleop_se3_agent.py`` and ``isaaclab teleop run`` exiting with status 0 when the environment or the
  teleop device could not be created. The original exception and traceback are now propagated.
