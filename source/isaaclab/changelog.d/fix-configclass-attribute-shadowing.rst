Fixed
^^^^^

* Fixed ``from isaaclab.utils import configclass`` returning the :mod:`isaaclab.utils.configclass`
  sub-module instead of the :func:`~isaaclab.utils.configclass.configclass` decorator. Once anything
  imported the sub-module first, the import machinery shadowed the lazily attached decorator on
  :mod:`isaaclab.utils` and ``@configclass`` failed with ``TypeError: 'module' object is not callable``.
