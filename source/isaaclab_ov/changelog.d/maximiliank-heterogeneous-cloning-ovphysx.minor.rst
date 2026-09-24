Added
^^^^^

* Added OvPhysX 0.6.3 fast-path cloning for heterogeneous rigid-body and articulation geometry variants while preserving per-environment collision grouping.
* Added validation of variant body/joint connectivity and effective D6 axes before cloning.

Fixed
^^^^^

* Fixed contacts between retained source variants and cloned assets, collision isolation, and environment-indexed tensor reads and writes in heterogeneous scenes.
* Preserved cloned contact reporters and kept sensor rows and resolved contact filters in environment order.
* Removed authored runtime clone targets from retained source environments and preserved the legacy homogeneous clone signature on older OvPhysX versions.
* Registered Newton USD schemas before loading OvStage scenes so cloned mimic-joint constraints work.
* Reused resolved articulation and rigid-body paths to accelerate tensor binding in large scenes.
