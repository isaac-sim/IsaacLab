* Added opt-in OVStage sharing between OVPhysX and OVRTX through ``ISAAC_LAB_SHARE_OVSTAGE=1``. Clone preparation combined
  their asset routes and populated both USD domains before either consumer attached. Without flags, OVPhysX
  retained native physics cloning with independent OVStage rendering, and Newton retained native OVRTX cloning.
  Sharing remained opt-in pending OVPhysX binding improvements.
* Shared one OVRTX engine across camera configurations on a shared physics stage and submitted all its active
  products. Cameras must agree on native logging and transform-cache settings and precede the first reset.
* Coordinated gravity control ordinals with rendering writes and rebuilt the shared stage from current USD
  during forced physics initialization after releasing bindings and detaching consumers.
  Private physics stages reused the same stage owner for population, queries, and write ordinals.
* Rejected full-stage physics imports in shared mode instead of dropping authored environment overrides.
  Those scenes must leave ``ISAAC_LAB_SHARE_OVSTAGE`` disabled.
* Removed custom import-error wrappers and redundant schema-registration fallbacks. Missing dependencies
  retained Python's original import errors; renderer cleanup propagated binding failures after releasing resources.
