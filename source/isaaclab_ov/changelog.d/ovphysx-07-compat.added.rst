* Added opt-in OVPhysX 0.7 support while retaining the pinned and legacy runtime paths,
  including task completion, stage ownership, external schema discovery and renderer scene reads.
* Preserved legacy joint velocity and armature settings in private OVPhysX 0.7 exports.
  Unsupported nonzero legacy joint friction now reports a migration error instead of being silently ignored.
