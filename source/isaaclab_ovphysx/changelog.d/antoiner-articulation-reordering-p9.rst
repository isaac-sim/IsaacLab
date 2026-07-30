Added
^^^^^

* Added OvPhysX articulation Jacobians, mass matrices, and gravity compensation
  through the backend-agnostic articulation data API.

Changed
^^^^^^^

* Cached stable OvPhysX articulation read launches outside CUDA graph capture.
  No user migration is required.
