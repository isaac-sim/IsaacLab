Changed
^^^^^^^

* Changed the SO101 keyboard reset buffer to cache one snapshot per environment by default, so its first
  IK batch uses every environment. Set ``commands.typing.reset.buffer_size`` to a positive integer to
  retain an explicit capacity; ``8192`` restores the previous size. Partial batches were updated to sample
  distinct environments across the full scene instead of only its first clone variants.
