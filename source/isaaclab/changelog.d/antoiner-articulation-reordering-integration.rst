Added
^^^^^

* Added an opt-in cached :class:`~isaaclab.utils.warp.ProxyArray` return mode
  to asset finder methods for zero-copy Torch and Warp index views.

Deprecated
^^^^^^^^^^

* Deprecated relying on the implicit legacy return type of asset finder
  methods. Pass ``as_proxy=True`` for cached proxy selectors or
  ``as_proxy=False`` to retain the current legacy representation explicitly.

Fixed
^^^^^

* Fixed shared articulation ordering and external wrench paths to accept signed
  32-bit and signed 64-bit selectors without allocating Torch conversion tensors.
