* Added start-up status lines to the OVXR application log: a progress line every 30 s while the renderer
  attaches the scene, and a ready line when it finishes. A first start compiles shaders for minutes
  without logging, which looked like a hang, and no client can connect until attachment ends.
