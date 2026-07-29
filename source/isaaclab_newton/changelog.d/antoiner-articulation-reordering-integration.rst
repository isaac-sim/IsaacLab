Added
^^^^^

* Added cached :class:`~isaaclab.utils.warp.ProxyArray` selector support to
  Newton asset finder methods.

Changed
^^^^^^^

* Cached stable articulation read launches outside CUDA graph capture on
  Newton. No user migration is required.

Deprecated
^^^^^^^^^^

* Deprecated relying on implicit legacy finder returns. Pass
  ``as_proxy=True`` for cached proxy selectors or ``as_proxy=False`` to retain
  the current legacy representation explicitly.

Fixed
^^^^^

* Fixed Newton indexed articulation writes to accept signed 32-bit and signed
  64-bit selectors without allocating a Torch conversion tensor.
* Fixed stale pose-, velocity-, and center-of-mass-derived rigid asset data
  immediately after simulation state and property writes.
