Added
^^^^^

* Added cached :class:`~isaaclab.utils.warp.ProxyArray`
  selector support to Newton asset finder methods.

Deprecated
^^^^^^^^^^

* Deprecated relying on implicit legacy finder returns. Pass ``as_proxy=True`` for cached proxy selectors or
  ``as_proxy=False`` to retain the current legacy representation explicitly.
