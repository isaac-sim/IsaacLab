* Switched supported contributed task defaults from Isaac Sim PhysX and RTX to OvPhysX and OVRTX,
  preserving explicit scene tuning and the original policy-facing articulation axes. These defaults require the ``ov`` installation extra.
* Updated the coffee demo to open its Newton RTX kitchen view by default; ``--visualizer newton_gl`` selects
  the lightweight OpenGL tabletop view. RTX requires the ``ovrtx`` installation extra.
* Retained explicit Isaac RTX camera presets and the three-argument Factory friction helper for legacy backend selections.
