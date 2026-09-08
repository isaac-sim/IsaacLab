Added
^^^^^

* Added stable environment IDs to OVPhysX GPU clone recipes whose sources all reside in env 0.
  CPU, full-stage, and nonzero-source layouts instead materialize the clone plan and author USD
  collision groups, preserving same-environment contact and cross-environment isolation.
* Added an explicit error for declarative manager collision groups, which OVPhysX does not yet
  support, instead of silently applying a weaker policy.
