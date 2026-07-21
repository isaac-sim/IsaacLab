Added
^^^^^

* Added an opt-in cached :class:`~isaaclab.utils.warp.ProxyArray`
  return mode to asset finder methods for
  zero-copy Torch and Warp index views.

Deprecated
^^^^^^^^^^

* Deprecated relying on the implicit legacy return type of asset finder
  methods. Pass ``as_proxy=True`` for the future cached proxy behavior or
  ``as_proxy=False`` to retain the current legacy representation explicitly.
