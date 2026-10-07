* **Breaking:** Replaced Newton GL's floating camera panel with a selectable main view.
  Used the first configured ``cameras`` source initially, or perspective mode when none was configured.
  Selected scene-camera views through the sidebar's Camera View dropdown instead of the old Open/Hide toggle.
  Navigation applied the same camera-local motion to every copy of the selected sensor;
  unselected sensors were neither read nor moved by the viewer.
* Used scene-authored lights in Newton RTX by default, including HDR textures, intensity, initial
  transforms, and clone placements. An explicit solid background preserved scene illumination.
* Replaced the separate Newton RTX scene importer with simulation-owned OVRTX rendering.
  Added scene-camera selection and device-image presentation to RTX; resizing and closing the
  perspective view left shared sensor products unchanged.
* Preserved RTX debug markers and visibility controls through shared renderer-owned geometry and GPU pose updates.
* Presented Newton GL and RTX camera views directly from composed device images. Recording and web
  transports read back only the composed image.
* **Breaking:** Removed Newton RTX's viewer-only ``rtx_environment`` and ``world_spacing`` overrides
  and GL model options. Declare lighting, materials, and environment placement in the scene instead.
  Use Newton GL for rigid-body dragging and model overlays. Environment selection now limited RTX
  sensor display tiles; its perspective camera viewed the full shared scene.
