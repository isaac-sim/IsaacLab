Added
^^^^^

* Added the per-modality camera observation terms :class:`~isaaclab.envs.mdp.observations.image_rgb`,
  :class:`~isaaclab.envs.mdp.observations.image_depth`,
  :class:`~isaaclab.envs.mdp.observations.image_normals` and
  :class:`~isaaclab.envs.mdp.observations.image_segmentation`. Each supports ``channel_first`` and
  ``frame_stack``, and works with :class:`~isaaclab.sensors.Camera` and
  :class:`~isaaclab.sensors.RayCasterCamera` outputs.
* Added :func:`~isaaclab.utils.images.normalize_rgb`, :func:`~isaaclab.utils.images.normalize_depth`,
  :func:`~isaaclab.utils.images.normalize_normals` and
  :func:`~isaaclab.utils.images.normalize_segmentation`. ``normalize_rgb`` accepts a constant
  ``mean``, and ``normalize_depth`` supports ``invalid_value``, ``max_depth`` and ``tanh_scale``.
* Added :class:`~isaaclab.utils.images.CameraFrameStack`, the shared layout, frame-stacking and
  deferred-normalization pipeline for camera observations in manager-based and direct environments.

Changed
^^^^^^^

* Moved the fused uint8 normalization kernels from :mod:`isaaclab.utils.warp.kernels` into
  :mod:`isaaclab.utils.images`, so all camera-image normalization lives in one module.
* Changed depth normalization in :func:`~isaaclab.utils.images.normalize_camera_image` to also replace
  NaN and negative infinity with zero, and to return a new tensor instead of modifying the camera
  buffer in place.

Deprecated
^^^^^^^^^^

* Deprecated :class:`~isaaclab.envs.mdp.observations.stacked_image`. Use the ``frame_stack`` parameter
  of the per-modality image terms.
* Deprecated :func:`isaaclab.utils.warp.ops.normalize_image_uint8`. Use
  :func:`~isaaclab.utils.images.normalize_rgb`, which takes the same arguments.

Removed
^^^^^^^

* **Breaking:** Removed ``isaaclab.envs.mdp.image``. Use ``image_rgb``, ``image_depth``,
  ``image_normals`` or ``image_segmentation`` in observation term configs; replace ``permute=True``
  with ``channel_first=True`` and remove ``clone``. For direct tensor processing, use the
  normalizers and ``CameraFrameStack`` in :mod:`isaaclab.utils.images`. The new terms return
  independent tensors; ``image_rgb`` drops alpha channels.
