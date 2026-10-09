* Inlined camera render specifications at initialization so renderer inputs use the existing simulation context
  and remain visible at their call sites.
* Kept renderer stage preparation in the camera lifecycle instead of registering an empty preparation hook
  for every sensor. All cameras still apply USD overrides before the shared stage is exported.
