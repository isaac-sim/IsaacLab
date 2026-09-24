Added
^^^^^

* Added a ``stationary`` option to :func:`~isaaclab.envs.mdp.observations.image` and
  :func:`~isaaclab.utils.images.normalize_camera_image` that maps RGB-like and depth-like images
  to a fixed ``[-0.5, 0.5]`` range, independent of per-frame statistics.
* Added support for ``data_type=None`` in :func:`~isaaclab.envs.mdp.observations.image` to select
  the data type of a single-output camera.
* Added ``output_channel_dim`` to :func:`~isaaclab.utils.images.normalize_camera_image` and
  :func:`~isaaclab.utils.warp.ops.normalize_image_uint8` to convert between BHWC and BCHW while
  normalizing, and ``center`` to :func:`~isaaclab.utils.warp.ops.normalize_image_uint8` to subtract
  a constant instead of the per-image mean.

Changed
^^^^^^^

* Changed the default of ``clone`` in :func:`~isaaclab.envs.mdp.observations.image` to False,
  because the observation manager already copies every term's output. Direct callers that mutate
  an unnormalized result should pass ``clone=True``.
* Reduced camera observation overhead by permuting normalized images in the normalization kernel,
  skipping the camera mask device-to-host copy when ``update_period`` is zero, and skipping the
  concatenation copy for single-term observation groups without history.
