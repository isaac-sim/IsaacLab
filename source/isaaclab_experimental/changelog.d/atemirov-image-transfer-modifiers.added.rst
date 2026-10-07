* Added :mod:`isaaclab_experimental.image_transfer` for application-owned image generation from camera outputs as
  modifiers: :class:`~isaaclab_experimental.image_transfer.ImageTransferModifierCfg` with chunked per-view queues,
  per-view resets with deterministic seeds, and a model shared through the simulation context, plus the
  ``depth_to_control`` and ``srgb_to_linear`` function modifiers.
