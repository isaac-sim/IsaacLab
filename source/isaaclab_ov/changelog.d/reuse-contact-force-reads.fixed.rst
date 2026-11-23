* Avoided redundant OvPhysX contact-force reads when lazy sensor processing followed an eager
  refresh on the same physics step and stream, while retaining per-step contact-loss updates.
