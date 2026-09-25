Added
^^^^^

* Added support for ``data_type=None`` in :func:`~isaaclab.envs.mdp.observations.image` to select
  the data type of a single-output camera.
* Added ``output_channel_dim`` to :func:`~isaaclab.utils.images.normalize_camera_image` and
  :func:`~isaaclab.utils.warp.ops.normalize_image_uint8` to convert between BHWC and BCHW while
  normalizing.
* Added :attr:`~isaaclab.managers.ObservationTermCfg.clone_output` to skip the observation manager's
  copy for terms that return a new tensor on every call, such as normalized images.

Changed
^^^^^^^

* Changed the default of ``clone`` in :func:`~isaaclab.envs.mdp.observations.image` to False,
  because the observation manager already copies every term's output. Direct callers that mutate
  an unnormalized result should pass ``clone=True``.
* Changed :func:`~isaaclab.utils.warp.ops.normalize_image_uint8` to accept strided input, such as
  the RGB view of an RGBA camera buffer, so camera images use the fused normalization kernel
  instead of a multi-pass PyTorch fallback.
* Reduced camera environment overhead by skipping the camera mask device-to-host copy when
  ``update_period`` is zero, skipping the concatenation copy for single-term observation groups
  without history, and removing per-step host synchronizations from the termination manager and
  from visualization-marker index validation.

Fixed
^^^^^

* Fixed :func:`~isaaclab.envs.mdp.observations.image` passing the sensor's ``ProxyArray`` to the
  image normalization, which left colorized semantic segmentation unscaled in ``[0, 255]`` and
  skipped the fused normalization kernel.
