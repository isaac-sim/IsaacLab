* Changed ``newton_rtx`` to display simulation-owned OVRTX images in a Newton GL window, eliminating its separate renderer and stage. Preserved fixed-resolution capture, scene-camera selection, markers, and runtime surface meshes.
* Added ``NewtonRTXVisualizerCfg.renderer_cfg``. Match camera sensors' ``OVRTXRendererCfg`` to share their renderer; default configurations share automatically.
