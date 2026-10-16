Fixed
^^^^^

* Fixed :func:`~isaaclab.utils.configclass` on Python 3.14, where lazily evaluated annotations (PEP 649) are no
  longer stored in the class ``__dict__`` and every config class without ``from __future__ import annotations``
  failed with a mismatch between annotations and class members.
