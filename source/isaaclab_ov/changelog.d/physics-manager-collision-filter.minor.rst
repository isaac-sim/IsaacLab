Added
^^^^^

* Added manager-owned replicated-environment isolation, using stable native environment IDs across
  GPU clone calls and validated USD groups for full-stage fallback. Declarative semantic groups and
  fallback composition with unrelated authored groups fail explicitly because OVPhysX does not yet
  support their safe lowering.
