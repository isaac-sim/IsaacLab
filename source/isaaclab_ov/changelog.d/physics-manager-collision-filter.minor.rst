Added
^^^^^

* Added stable environment IDs to OVPhysX clone recipes so heterogeneous assets in the same
  replicated environment retain contact while cross-environment collisions stay isolated.
* Added an explicit error for declarative manager collision groups, which OVPhysX does not yet
  support, instead of silently applying a weaker policy.
