* Added :func:`~isaaclab_experimental.cosmos.cosmos_camera` and :func:`~isaaclab_experimental.cosmos.apply_cosmos`
  to put Cosmos on compatible RGB-only pinhole cameras at a Cosmos canvas with the same view and original output size, with
  :func:`~isaaclab_experimental.cosmos.service_capabilities`,
  :func:`~isaaclab_experimental.cosmos.service_max_episode_frames` and ``COSMOS_CANVASES``,
  plus :func:`~isaaclab_experimental.image_transfer.center_crop_resize`.
* With several environments, :func:`~isaaclab_experimental.cosmos.apply_cosmos` requires a camera that captures
  every environment step, since independent resets would otherwise send extra frames for the environments a
  capture skipped.
