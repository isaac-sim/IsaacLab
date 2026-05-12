Deprecated
^^^^^^^^^^

* Deprecated :attr:`~isaaclab.scene.InteractiveSceneCfg.clone_in_fabric`.
  The field is currently a no-op — no physics backend (PhysX, Newton,
  OmniPhysX) reads its value, so setting it to ``True`` has no effect.
  Setting the field to ``True`` now raises a :class:`DeprecationWarning`.
  Remove the kwarg from your :class:`~isaaclab.scene.InteractiveSceneCfg`
  constructor; the field will be removed in a future release.
