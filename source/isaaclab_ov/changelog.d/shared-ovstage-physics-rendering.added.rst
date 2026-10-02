* Added opt-in OVStage sharing between OVPhysX and OVRTX through ``ISAAC_LAB_SHARE_OVSTAGE=1``, equivalent to
  enabling both ``ISAAC_LAB_OVRTX_USE_OVSTAGE`` and ``ISAAC_LAB_OVPHYSX_USE_OVSTAGE``. Clone preparation combined
  their asset routes and populated both USD domains before either consumer attached. Without flags, OVPhysX
  retained native physics cloning with independent OVStage rendering, and Newton retained native OVRTX cloning.
  Sharing remained disabled pending OVPhysX binding improvements; conflicting flags raised configuration errors.
* Shared one OVRTX engine across camera configurations on a shared physics stage and submitted all its active
  products. Cameras must agree on native logging and transform-cache settings and precede the first reset.
* Coordinated gravity control ordinals with rendering writes and rebuilt the shared stage from current USD
  during forced physics initialization after releasing bindings and detaching consumers.
