Added
^^^^^

* Added ``output_channel_dim`` to :func:`~isaaclab.utils.images.normalize_camera_image` and
  :func:`~isaaclab.utils.warp.ops.normalize_image_uint8` to convert between BHWC and BCHW while
  normalizing.

Changed
^^^^^^^

* Changed :func:`~isaaclab.utils.warp.ops.normalize_image_uint8` to accept strided input, such as
  the RGB view of an RGBA camera buffer, and to write its output in coalesced order, so camera
  images use the fused normalization kernel instead of a multi-pass PyTorch fallback.
* Changed :func:`~isaaclab.envs.mdp.observations.image` and the Cartpole camera observations to
  permute normalized images to channel-first inside the normalization kernel.
