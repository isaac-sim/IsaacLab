* Reduced shared OVStage initialization work and host memory by attaching and cloning physics originals
  before acquiring bindings and completing render copies. This order is selected automatically with
  ``ISAAC_LAB_SHARE_OVSTAGE=1``; shared stages remain experimental and opt-in.
* Completed shared-stage bindings and render copies before public physics-readiness callbacks, preserving
  startup gravity edits. Failed render copies remained pending and required a stage rebuild before retry.
* Preserved full-stage physics import and world placement for CPU simulations using a shared OVStage.
* Exposed the OVPhysX binding phase through ``binding_init_event`` and ``initialize_event`` attributes,
  removing asset and contact-sensor dependencies on private event selection.
