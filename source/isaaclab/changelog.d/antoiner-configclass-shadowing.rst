Fixed
^^^^^

* Fixed ``from isaaclab.utils import configclass`` returning the submodule instead of the decorator
  after any import of ``isaaclab.utils.configclass`` (order-dependent
  ``TypeError: 'module' object is not callable``). The submodule is now callable and forwards to
  the decorator, so both import spellings work regardless of import order.
