Fixed
^^^^^

* Fixed ``from isaaclab.assets import Articulation`` and ``from isaaclab.sim import SimulationContext``
  loading ``pxr`` at module-import time, which forced :class:`~isaaclab.app.AppLauncher`
  to run before any such import and blocked kit-less workflows that bind these
  classes at module top.
