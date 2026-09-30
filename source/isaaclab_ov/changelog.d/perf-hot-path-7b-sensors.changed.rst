* Changed OVPhysX PVA and frame-transformer debug visualization to refresh outdated sensor
  buffers before drawing.
* Changed the OVPhysX rigid-object ``body_com_acc_w`` finite difference to use the elapsed time since the
  previous update, matching joint accelerations and the Newton backend.

* Skipped the joint-limit clamping counter readback when its logging level is disabled and reused
  the counter buffer across writes.
