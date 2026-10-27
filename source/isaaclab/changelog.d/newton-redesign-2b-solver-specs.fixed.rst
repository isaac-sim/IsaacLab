* Fixed :func:`~isaaclab.utils.configclass` turning a class variable inherited from a configuration declared with
  ``from __future__ import annotations`` into a field, when the subclass's module does not import ``ClassVar``.
