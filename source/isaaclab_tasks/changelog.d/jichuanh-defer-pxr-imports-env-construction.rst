Fixed
^^^^^

* Fixed eager module-level ``pxr``/``carb`` imports in the cabinet and factory
  direct environments that could bind a mismatched USD/Carbonite build before Kit
  launches; the imports are now deferred into the methods that use them. Extended
  the env-config import guard test to also import each task's env-class entry point,
  so this class of eager backend import is caught for the ``gym.make`` path too.
