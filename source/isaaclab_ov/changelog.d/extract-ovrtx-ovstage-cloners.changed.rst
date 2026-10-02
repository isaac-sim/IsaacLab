* Moved clone-plan interpretation out of OVRTX camera initialization and into its clone context. Both scene
  paths consumed the same prepared copies after camera overrides were authored. The renderer cloned and
  exported only assets routed to OVRTX, including spawned and shared assets.
* Unified first and subsequent OVRTX camera registration, and propagated object scales from the prepared
  clone copies instead of traversing the clone plan again.
* Consolidated scene binding setup, camera pose updates, and render submission across OVRTX scene paths.
  Converted camera poses directly into retained matrix buffers and shared unscaled SDP publications between renderers.
  Paired query and path-list lifetimes on the stage backend, including material updates, and decoded segmentation
  labels once per frame. Derived active products from registered cameras.
* Shared one OVRTX engine across camera product configurations when opting into a shared OVPhysX stage, as required
  by the SDK's single-renderer attachment contract. Cameras must agree on native logging and transform-cache settings.
  Removed duplicate USD scene-partition authoring; camera registration authored the runtime attributes.
* Created native environment frames only in private USD exports and placed OVPhysX originals from the clone plan.
  Replaced recursive export filtering with a pruned USD traversal while retaining routed prototype selection.
* Moved shared-stage selection from renderer construction into clone preparation. OVRTX configurations
  requested native cloning by default; preparation replaced the physics/render routes when explicitly
  opting into OVStage and acquired their resources only after deciding scene ownership.
