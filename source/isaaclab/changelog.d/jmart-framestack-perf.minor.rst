Added
^^^^^

* Added :paramref:`~isaaclab.utils.buffers.CircularBuffer.stack_dim` constructor argument
  and :attr:`~isaaclab.utils.buffers.CircularBuffer.stacked` property: when ``stack_dim`` is
  set, the internal storage is rearranged so ``stacked`` returns the ``K`` frames merged
  along the chosen dim as a free contiguous view.
* Added :mod:`isaaclab.utils.images` with :func:`~isaaclab.utils.images.normalize_camera_image`
  and the ``is_rgb_like`` / ``is_depth_like`` / ``is_normals_like`` predicates, shared
  between the DirectRLEnv and ManagerBasedEnv camera observation paths.
* Added :func:`isaaclab.utils.warp.ops.normalize_image_uint8`, a fused Warp-kernel
  implementation of ``(uint8 / 255) - per-image-channel mean`` for RGB-like camera
  observations. Supports both ``(B, H, W, C)`` and ``(B, C, H, W)`` inputs via a
  ``channel_dim`` argument (``-1`` / ``3`` for BHWC, ``-3`` / ``1`` for BCHW); the
  argument is also forwarded by :func:`~isaaclab.utils.images.normalize_camera_image`.
* Added a ``clone`` kwarg to :func:`isaaclab.envs.mdp.observations.image`; callers that
  immediately copy the result into their own storage (e.g. a frame-stack buffer) can pass
  ``clone=False`` to skip the redundant allocation.

Changed
^^^^^^^

* Changed :class:`~isaaclab.envs.mdp.observations.stacked_image` to use the new ``stack_dim``
  ``CircularBuffer`` layout and defer normalization past the frame-stack buffer for RGB-like
  data types, eliminating a per-frame float32 upcast and large transpose.
