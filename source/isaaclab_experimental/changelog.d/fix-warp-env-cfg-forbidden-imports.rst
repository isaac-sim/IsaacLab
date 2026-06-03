Fixed
^^^^^

* Fixed the experimental packages eagerly importing backend modules (``pxr``,
  ``omni``, ``carb``, ``isaacsim``, ``scipy``) at import time, which crashed when
  a warp task's env config was loaded before ``SimulationApp`` was launched. The
  ``managers``, ``envs``, ``envs.mdp`` and ``envs.mdp.actions`` packages now use
  ``lazy_export`` with ``.pyi`` stubs, and the MDP term leaf modules guard runtime
  types (``Articulation``, ``InteractiveScene``, ``ContactSensor``, action terms)
  under ``TYPE_CHECKING`` with string ``class_type`` references.
