* **Breaking:** Replaced Newton GL's floating camera panel with a selectable main view.
  Used the first configured ``cameras`` source initially, or perspective mode when none was configured.
  Selected scene-camera views through the sidebar's Camera View dropdown instead of the old Open/Hide toggle.
  Navigation applied the same camera-local motion to every copy of the selected sensor;
  unselected sensors were neither read nor moved by the viewer.
* Used scene-authored lights in Newton RTX by default, including HDR textures, intensity, initial
  transforms, and clone placements. An explicit solid background preserved scene illumination.
* Used Newton's native ``ViewerRTX`` with a simulation-owned OVStage and preserved its debug markers.
  Added scene-camera selection and device-image presentation to RTX. Window resizing scaled the fixed
  perspective image; resizing and closing the window left camera sensors unchanged.
  Backend initialization resolved native resources from the explicitly supplied simulation owner;
  core initialization and reset no longer passed backend-specific resources.
  Pinned Newton to the merged borrowed-stage implementation pending its next release.
* Presented Newton GL and RTX camera views directly from composed device images. Recording and web
  transports read back only the composed image.
* Unified Newton GL window and headless frame rendering, and propagated rendering errors after frame
  cleanup. Kept contact-sensor arrow data on the viewer device during presentation.
  Corrected PhysX contact arrow origins when displaying a subset of environments and removed
  wind controls that did not apply forces to the simulation.
* Read live-plot histories from Newton's plot logger, restoring scalar and array plots with the updated Newton API.
* **Breaking:** Removed Newton RTX's viewer-only ``rtx_environment`` and ``world_spacing`` overrides
  and GL model options. Declare lighting, materials, and environment placement in the scene instead.
  Use Newton GL for rigid-body dragging and model overlays. Environment selection now limited RTX
  sensor display tiles; its perspective camera viewed the full shared scene.
* **Breaking:** Replaced ``window_width`` and ``window_height`` with ``window=WindowCfg(size=(width, height))``
  for Newton and Kit. Replaced Newton's ``update_frequency`` frame skipping with ``window.fps``;
  the default 30 Hz presentation limit used wall-clock time and did not throttle simulation or headless capture.
