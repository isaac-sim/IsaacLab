Added
^^^^^

* Added :meth:`~isaaclab.utils.wrench_composer.WrenchComposer.resolve_submission`, which returns the
  cheapest representation of the buffered external wrench a consumer can accept, along with the
  :class:`~isaaclab.utils.wrench_composer.WrenchComposer.Frame` enum describing it. Consumers that can
  apply a world-frame wrench at the center of mass opt in with the new ``supports_world_at_com``
  constructor argument. Wrenches that are already local-frame, or already global-frame at the center of
  mass, are now submitted without composing them through the body poses.
