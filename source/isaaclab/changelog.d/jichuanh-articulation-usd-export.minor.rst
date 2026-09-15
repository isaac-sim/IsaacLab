Added
^^^^^

* Added complete fixed single-environment USD export through ``InteractiveScene.export_to_usd`` and asset-owned fixed-configuration authoring, with property-local USD bindings and a shared ``UsdWriter``, dependency/completeness checks, scalar-array and multi-axis joint authoring, and atomic saving. Selected one environment through ClonePlan queries and retained its initialized physical properties, placement, mass/inertia/COM and shared resources without resetting the source scene.
* Removed dangling direct material targets from exported copies only when all effective material bindings remained unchanged; retained errors for missing resources and bindings that affected inheritance.

* Added automatic scene export before startup events and fixed inertia offsets for rigid objects and collections.

* Excluded transient joint-state samples from deployment exports and zeroed initial velocities.
* Preserved source PhysX materials and automatic offsets in pre-startup deployment exports.
