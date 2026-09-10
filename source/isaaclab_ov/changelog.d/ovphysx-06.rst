Changed
^^^^^^^

* **Breaking:** Required OvPhysX 0.6.2 and OVStage 0.2.0.377349. Reinstall the ``ovphysx`` or ``ov`` extra with access to
  NVIDIA's internal Omniverse Artifactory index to obtain the matching runtime and OmniClient 2.74.0.

* Updated the OVRTX extra to 0.5.0.377615 to match OVStage 0.2. Reinstall the ``ovrtx`` or ``ov`` extra when upgrading.

Fixed
^^^^^

* Removed the obsolete reversed-joint sign correction from OvPhysX Jacobians, mass matrices, and gravity compensation
  forces while preserving custom joint and body ordering.
* Registered the OvPhysX codeless physics schemas with OVStage before scene population, including when the host USD
  registry already provided those schemas.
