* Moved clone-plan interpretation out of OVRTX camera initialization and into its clone context. Both scene
  paths consumed the same prepared copies after camera overrides were authored. The renderer cloned and
  exported only assets routed to OVRTX, including spawned and shared assets.
* Unified first and subsequent OVRTX camera registration, and propagated object scales from the prepared
  clone copies instead of traversing the clone plan again.
* Consolidated scene binding setup, camera pose updates, and render submission across OVRTX scene paths.
  Converted camera poses directly into retained matrix buffers and shared unscaled SDP publications between renderers.
  Paired query and path-list lifetimes on the stage backend, including material updates, and decoded segmentation
  labels once per frame. Removed duplicate USD scene-partition authoring; camera registration authored
  the runtime attributes.
* Created native environment frames only in private USD exports and placed OVPhysX originals from the clone plan.
  Replaced recursive export filtering with a pruned USD traversal while retaining routed prototype selection.
* Moved scene selection from renderer construction into clone preparation. OVRTX configurations
  requested native cloning by default; preparation selected an isolated OVStage when explicitly
  requested and acquired resources only after deciding scene ownership.
* **Breaking:** Removed the OVStage 0.1 and OVRTX 0.4 compatibility modules and their version fallbacks.
  Used the pinned SDKs' GPU hierarchy model and RenderVar prim-path keys directly. Code importing
  ``ovstage_compat`` or ``renderers.ovrtx_compat`` must use OVStage 0.2 and OVRTX 0.5 APIs directly.
