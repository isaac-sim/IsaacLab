* Changed :meth:`~isaaclab.utils.wrench_composer.WrenchComposer.reset` to return without launching work
  when the composer is inactive, removing the per-step buffer clears for rigid objects and collections
  that apply no external wrench.
