Fixed
^^^^^

* Fixed :class:`~isaaclab.assets.AssetBase` and :class:`~isaaclab.assets.BaseArticulation`
  loading ``pxr`` at module-import time, which forced :class:`~isaaclab.app.AppLauncher`
  to run before any ``from isaaclab.assets import Articulation`` and blocked
  kit-less workflows that import asset classes at module top.
