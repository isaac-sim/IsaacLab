Fixed
^^^^^

* Fixed ``isaaclab.utils.configclass`` resolving to either the sub-module or the
  :func:`~isaaclab.utils.configclass.configclass` decorator depending on which one happened to be
  imported first. ``from isaaclab.utils import configclass`` could return the sub-module, making
  ``@configclass`` fail with ``TypeError: 'module' object is not callable``, and resolving the
  decorator first left the sub-module unreachable as an attribute of :mod:`isaaclab.utils`. The
  sub-module is now callable, so the decorator, ``import isaaclab.utils.configclass as ...`` and
  dotted attribute access all work regardless of import order. No migration is needed.
