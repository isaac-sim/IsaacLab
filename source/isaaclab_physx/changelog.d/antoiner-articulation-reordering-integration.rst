Added
^^^^^

* Added cached :class:`~isaaclab.utils.warp.ProxyArray` selector support to
  PhysX asset finder methods.

Changed
^^^^^^^

* Cached stable articulation read launches to reduce repeated Warp launch
  setup on PhysX. No user migration is required.

Deprecated
^^^^^^^^^^

* Deprecated relying on implicit legacy finder returns. Pass
  ``as_proxy=True`` for cached proxy selectors or ``as_proxy=False`` to retain
  the current legacy representation explicitly.

Fixed
^^^^^

* Fixed PhysX indexed articulation writes so joint, body, and tendon item
  selectors accept signed 32-bit and signed 64-bit integers without allocating
  Torch conversion tensors, while external environment indices remain signed
  32-bit.
* Fixed stale pose-, velocity-, and center-of-mass-derived rigid asset data
  immediately after simulation state and property writes.
* Fixed fixed and spatial tendon property writers to accept the selector
  arguments advertised by the common articulation interface.
* Fixed stale mass matrix and gravity compensation reads immediately after
  mass, inertia, and armature writes.
