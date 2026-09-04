Changed
^^^^^^^

* **Breaking:** Required class-based observation modifiers to use
  :class:`~isaaclab.utils.modifiers.ModifierBaseCfg`. Replace
  ``ModifierCfg(func=MyModifier, params=...)`` with ``ModifierBaseCfg(func=MyModifier, params=...)``. Function
  modifiers continue to use :class:`~isaaclab.utils.modifiers.ModifierCfg`.

Fixed
^^^^^

* Fixed function and class-based observation modifiers failing after a configuration dictionary round-trip.
