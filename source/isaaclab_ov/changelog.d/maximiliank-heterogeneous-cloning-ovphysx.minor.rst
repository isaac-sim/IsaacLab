Added
^^^^^

* Added OvPhysX 0.6.3 fast-path cloning for heterogeneous rigid-body and articulation geometry variants while preserving per-environment collision grouping.
* Added validation of variant body/joint connectivity and effective D6 axes before cloning.

Fixed
^^^^^

* Fixed contacts between retained source variants and cloned assets, collision isolation, and environment-indexed tensor reads and writes in heterogeneous scenes.
* Preserved cloned contact reporters and kept sensor rows and resolved contact filters in environment order.
* Removed authored runtime clone targets from retained source environments and preserved the legacy homogeneous clone signature on older OvPhysX versions.

Changed
^^^^^^^

* Updated the public extras to OvPhysX 0.6.3, OVStage 0.2.0.377349, and OVRTX 0.5.0.377615. Recreate or resync environments that use these extras so the compatible runtime wheels are updated together.
