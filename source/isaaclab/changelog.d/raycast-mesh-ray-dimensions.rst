Changed
^^^^^^^

* Changed :func:`~isaaclab.utils.warp.raycast_mesh` to raise a :class:`ValueError` unless
  ``ray_directions`` has the same shape as ``ray_starts``. Fewer directions than rays previously read
  past the end of ``ray_directions`` and returned undefined results, while surplus directions and
  differently shaped direction tensors were silently reinterpreted. Callers that relied on either
  must reshape ``ray_directions`` to match ``ray_starts``.

Fixed
^^^^^

* Fixed :func:`~isaaclab.utils.warp.raycast_mesh` shaping its distance and face-id outputs from only
  the first two ray dimensions. Whenever :attr:`return_distance` or :attr:`return_face_id` was
  enabled, rays with any shape other than ``(B, N, 3)`` either raised an error — including the
  documented ``(N, 3)`` input — or, when the leading dimensions happened to multiply out, silently
  returned a wrong-shaped tensor. Both outputs now keep the ray dimensions of the input.
