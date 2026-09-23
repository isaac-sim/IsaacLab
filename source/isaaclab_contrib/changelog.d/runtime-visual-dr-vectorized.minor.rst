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
* Changed the stacking demo configuration (now in ``isaaclab_tasks``) to a 640x480
  camera driving the segmentation hint, with
  ``aspect_ratio="4,3"``. Segmentation names regions instead of pinning geometry, so
  the background keeps the scene's layout and perspective while the prompt is free
  to populate it; the prompt bank is now a biological laboratory with researchers.

* Fixed ``PromptBankCfg.negative_prompt`` being declared but never sent to Cosmos.
  It now reaches the request, and the demo uses one to steer away from illustrated
  and rendered looks, which a positive prompt asking for realism competes with
  rather than excludes. The demo's prompts also describe photographic capture
  (camera, lens, lighting, grain) rather than just naming the scene.

* Added ``PromptBankCfg.progression`` and ``progression_steps``, a phrase sequence
  walked once across a run and appended to whichever variant the episode selected.
  It suits conditions that should drift rather than be drawn independently: the
  demo uses it to take the laboratory's windows from dawn to midnight over a
  rollout. The walk clamps at the final phrase rather than wrapping, so a longer
  run ends at midnight instead of snapping back to dawn.

* Added :class:`~isaaclab_contrib.visual_dr.RemoteCosmosBackendCfg`, which runs
  generation in worker processes on their own GPUs. It answers two problems at
  once: the simulator stops competing with a diffusion model for a GPU, and several
  workers generate the environments of one step at the same time -- the only
  parallelism available while Cosmos rejects batched transfer inference. Measured
  on H100s at 16 steps, one worker costs 3.62 s per frame, two 2.02 s and four
  1.18 s.

  Image payloads move by CUDA IPC and peer copy, so they never pass through host
  memory; only prompts, seeds and identifiers travel as ordinary objects. The
  runtime is unchanged: setting ``max_batch`` to the worker count makes it chunk
  the environments it wants randomized into exactly one frame per worker. Workers
  also run without Omniverse Kit, so the cuDNN attention fallback is inert there.

  Same-node only. Crossing machines needs a real transport behind the same class.

* Fixed ``VisualDRCfg.enabled`` being documented but never honored. The runtime
  constructed and activated a backend whatever the flag said, so a disabled
  configuration still loaded the model and still restyled frames, and the shipped
  default -- disabled, naming no backend -- could not be constructed at all. A
  disabled runtime now builds no backend, advances no scheduling state, and is
  indistinguishable from having attached no runtime: ``make_frame`` is never
  called, so depth and segmentation are not even fetched.

* Added ``CameraDRCfg.composite_foreground``. Set, the runtime pastes the preserved
  pixels back and the foreground is bit-identical to the render. Cleared, the
  generated frame stands as it is, letting the model light and render the foreground
  along with everything else: coherent lighting and no mask seam, but nothing
  constrains the objects' appearance -- in the stacking scene the cubes come back
  cyan and the robot black, which would be fatal for a policy that identifies a cube
  by colour. ``preserve_classes`` is required only while compositing, so a scene
  with no semantic tags works in the cleared mode.

* Added ``CosmosBackendCfg.mask_guidance``, which sends the preserved-foreground
  mask to Cosmos so it denoises around it rather than being pasted over blind.
  Stock ``cosmos-framework`` has no guided-generation support, so the backend
  detects the capability and refuses at load when it is requested and missing,
  instead of silently generating without a mask. Where it is available the
  generated background can agree with the foreground on lighting and contact
  shadows, which compositing alone cannot do.

* Fixed the ``cosmos-runtime`` extra being unsatisfiable alongside Isaac Lab. It
  pinned ``transformers<5`` and ``diffusers==0.35.1`` against Isaac Lab's own
  ``transformers==5.10.4``, so selecting it could not resolve. Both pins were
  stricter than reality: the transfer paths were verified on Python 3.12, torch
  2.12, transformers 5.10.4 and diffusers 0.39.0, with no flash-attn installed.
  Cosmos itself must still be installed with ``--no-deps`` -- its dependency groups
  pin torch to match the CUDA wheel variants it publishes -- and
  ``docs/visual_dr_setup.md`` records the recipe.

* Documented the setup gaps found in testing: the stacking task needs the ``teleop``
  extra, because the configuration imports ``XrCfg`` which now lives in
  ``isaaclab_teleop``; ``uv sync`` is exact and removes the Cosmos overlay along with
  around fifty packages unless ``--inexact`` is passed; and Cosmos cannot yet be
  locked alongside Isaac Lab, though only its ``transformers<5`` base pin stands in
  the way -- its base dependencies do not pin torch at all.

* Added ``scripts/visual_dr/install_cosmos.sh``, which installs a cosmos-framework
  checkout from a local path or an https or ssh git URL, optionally pinned with
  ``.git@<ref>``, always with ``--no-deps`` and alongside the ``cosmos-runtime``
  extra. It reports which checkout was installed and whether that checkout supports
  guided generation, so a checkout without it does not silently fall back.
* Added an early check that a checkpoint path exists. ``CosmosBackendCfg.checkpoint``
  already accepted a local directory as well as a registered name or ``s3://`` URI,
  but a mistyped path failed deep inside Cosmos with a message about config
  resolution rather than about the path.

* Documented that a custom Cosmos checkout may require FlashAttention and so may not
  run under Isaac Lab's Python and torch. Models using variable-length attention have
  no cuDNN path, and the Cosmos dependency index publishes FlashAttention only for
  CPython 3.13 against torch 2.9 or 2.10. The base ``Cosmos3-Nano`` path is unaffected,
  needing no FlashAttention at all.

* Documented how to unblock a custom Cosmos checkout that needs FlashAttention.
  Models using variable-length attention have no cuDNN path, and the Cosmos
  dependency index publishes FlashAttention only for CPython 3.13 against torch 2.9
  or 2.10. Dao-AILab's own releases carry a CPython 3.12 CUDA 13 build against torch
  2.10 which imports and runs on torch 2.12, varlen kernel included.

* Changed the stacking demo's prompts to describe the foreground as fixed -- a bare
  grey workbench holding a white arm and exactly three red, green and blue cubes --
  and added negatives against furnishing or recolouring it. This only affects the
  mode where ``composite_foreground`` is cleared; with the composite on those pixels
  come from the render and no prompt can reach them. Without it the model previously
  invented the foreground, returning fused cyan shapes in place of the cubes.

* Fixed the preserved mask being empty whenever ``composite_foreground`` was cleared.
  Mask construction was gated on the composite, so a backend sending the mask to the
  model as a guidance signal received an all-zero mask and the guidance silently did
  nothing. The mask is now built whenever ``preserve_classes`` names anything, and
  the composite flag decides only whether the runtime pastes it back. With a real
  mask, guided generation preserves the workbench colour exactly and keeps the cubes'
  colours while still lighting the whole frame coherently.
