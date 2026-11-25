* **Breaking:** Switched supported contributed task defaults from Isaac Sim PhysX and RTX to OvPhysX and OVRTX.
  Install the ``ov`` extra (``uv run --extra ov`` or ``isaaclab[ov]``) for these defaults. Select
  ``physics=isaacsim_physx`` to retain the supported legacy physics path; cameras automatically select its
  compatible RTX runtime. Existing controls, timing, scene tuning and policy-facing articulation axes are preserved.
* Updated the coffee demo to open its Newton RTX kitchen view by default; ``--visualizer newton_gl`` selects
  the lightweight OpenGL tabletop view. RTX requires the ``ovrtx`` installation extra.
* Retained the three-argument Factory friction helper for legacy backend selections.
