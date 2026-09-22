Added
^^^^^

* Added a configuration schema for runtime visual domain randomization
  (:class:`~isaaclab_contrib.visual_dr.VisualDRCfg`) covering preserved foreground
  classes, restyle probability and scope, style persistence, action-chunk gating and
  the failure policy, with backends selected through ``class_type``.
* Added a Cosmos transfer backend that runs against a stock ``cosmos-framework``
  checkout and the public ``Cosmos3-Nano`` checkpoint. Foreground preservation is a
  composite rather than mask-guided denoising, which is what removes the need for a
  patched checkout.
* Added a passthrough backend for bringing a task up before a model is available.

  The Cosmos backend generates one frame per call, because stock
  ``cosmos-framework`` rejects batched transfer inference. Randomizing N
  environments therefore costs N sequential calls until multi-sample control
  packing lands upstream in Cosmos.

Changed
^^^^^^^

* Changed the visual DR runtime to operate on all environments at once. Episode
  boundaries, action-chunk gating and partial resets are now read from the
  environment rather than declared by the application, so ``DRObservation`` and
  ``ActionChunkSchedule`` are no longer needed and the application only owns model
  residency. Style and restyle decisions derive from a hash of the seed, environment
  index and episode index, so they are stable across replicas and repeated reads.
* Changed the preserved-region configuration to name foreground classes rather than
  replaceable background classes. Unrecognized semantic IDs are preserved by default,
  so an untagged asset survives generation instead of dissolving into the background.

Fixed
^^^^^

* Fixed Cosmos generation failing inside a running Omniverse Kit process. For
  128-wide attention heads cuDNN answers through a runtime-compiled engine, and
  once Kit has started, compiling a shape it has not already seen fails with
  ``mha_graph.execute(...)`` returning false. Sequence lengths follow the prompt,
  so no fixed warm-up covers them. The Cosmos backend now probes cuDNN with an
  unseen shape on load and, when it cannot compile, routes attention through
  torch's Flash kernels, which need no runtime compilation and return the same
  logsumexp statistics.
* Fixed the preserved-region mask against the RTX renderer, which reports
  ``idToLabels`` keys as ``"(r, g, b, a)"`` strings even when
  ``colorize_semantic_segmentation`` is disabled and the buffer holds uncolored
  signed ``int32`` IDs. Those channels are the little-endian bytes of the stored
  ID, and both key forms are now accepted.
* Fixed offload freeing no GPU memory. Cosmos' checkpoint loader keeps a state
  dict whose tensors share storage with the model's parameters, so moving the
  parameters to host left the storages live and the next activation allocated a
  second copy -- residency grew by the size of the model on every cycle, which
  defeats offloading. The backend now releases that retained copy, matching it by
  storage address, and an offload/activate cycle returns to its starting residency
  exactly.

* Lowered the default Cosmos sampler steps from four to two, roughly halving
  generation latency (0.94 s to 0.49 s per frame on an H100 at 200x200). Cost is
  close to linear in steps, so the setting is the main latency control;
  ``scripts/visual_dr/bench_steps.py`` sweeps it against appearance on a real
  simulated frame.

* Fixed the demo rendering heavily speckled images. The Cosmos visuomotor config it
  derives from sets ``antialiasing_mode="Off"``, and on a data-center GPU that means
  no denoising at all -- DLSS Ray Reconstruction is unavailable on H100, so RTX
  Real-Time returns raw single-sample output. The demo now selects ``DLAA``, which
  denoises without depending on ray reconstruction, and re-renders on reset, both
  matching the non-Cosmos visuomotor task.

* Changed the demo to randomize every consumed observation (``probability`` 1.0),
  widened its table camera to roughly an 82 degree field of view and pushed the far
  plane from 2 m to 50 m, and replaced the prompt bank with crowded conference and
  auditorium scenes. The task frames its cameras tightly on the table with a 2 m far
  plane, which left almost nothing for a background prompt to fill.
* Added ``CosmosBackendCfg.depth_range_m``. The control map previously normalized
  against each frame's own depth extremes, so widening the far plane collapsed the
  near geometry to a couple of dark values and destroyed the control signal.

* Fixed washed-out, generic backgrounds. The Cosmos manifest was built with
  ``guidance`` 1.0, which effectively disables classifier-free guidance, so the
  prompt had little influence on the result. Cosmos' own ``image2image`` defaults
  are ``guidance`` 6.0 with 35 steps; the backend now defaults to 6.0, and
  ``num_steps`` rises from two to sixteen because high guidance with very few steps
  produces hard contrast rather than detail.
* Changed the demo to randomize only the table camera. The wrist camera sits
  centimetres from the table and barely sees the room, so generating for it cost a
  frame per step and changed almost nothing.

* Fixed ``control_kind`` selecting a hint name without changing the signal sent.
  Any value produced a depth-derived map, so ``"seg"`` silently mislabelled depth
  data as segmentation. Segmentation is now derived from the renderer's own buffer,
  carried on :class:`~isaaclab_contrib.visual_dr.DRFrame`; the semantic IDs are the
  little-endian bytes of the colour the renderer would have drawn, so unpacking
  recovers that palette exactly. Unimplemented hints are rejected rather than
  quietly substituted.
* Changed the prompt-facing defaults to Cosmos' tuned transfer values --
  ``guidance`` 3.0 and ``control_guidance`` 1.5 -- rather than the ``image2image``
  defaults, which are for the mode that has no control hint. Cosmos tunes
  ``control_guidance`` per hint (depth 1.5, seg 2.0, edge/blur 1.5, wsm 3.0) and
  applies those only when the request omits the field, which this backend never does.
* Changed the demo to a 640x480 camera driving the segmentation hint, with
  ``aspect_ratio="4,3"``. Segmentation names regions instead of pinning geometry, so
  the background keeps the scene's layout and perspective while the prompt is free
  to populate it; the prompt bank is now a biological laboratory with researchers.

* Fixed ``PromptBankCfg.negative_prompt`` being declared but never sent to Cosmos.
  It now reaches the request, and the demo uses one to steer away from illustrated
  and rendered looks, which a positive prompt asking for realism competes with
  rather than excludes. The demo's prompts also describe photographic capture
  (camera, lens, lighting, grain) rather than just naming the scene.
