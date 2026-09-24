Changed
^^^^^^^

* Published OVPhysX rigid poses directly into shared scene-data storage and invalidated cached transforms after
  physics steps and manual pose writes. Binding failures were surfaced instead of publishing incomplete poses.
* Routed OVRTX rigid transforms through cached SDP matrix requests, preserving authored scales without a Newton
  rigid-state intermediary. Existing renderer configurations remained valid; Newton-backed deformable, particle,
  and cable geometry transport remained unchanged.
* Captured OVRTX authored scales from clone-plan prototypes and shared roots, including bodies outside the
  default environment namespace.
