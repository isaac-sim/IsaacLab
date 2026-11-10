* Reduced shared OVStage initialization work and host memory by attaching and cloning physics originals
  before acquiring bindings and completing render copies. This order is selected automatically with
  ``ISAAC_LAB_SHARE_OVSTAGE=1``; shared stages remain experimental and opt-in.
