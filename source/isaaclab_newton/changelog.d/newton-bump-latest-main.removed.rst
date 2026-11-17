* **Breaking:** Removed ``VBDSolverCfg.rigid_compliant_alm``. Newton 1.9 always solves VBD rigid joints and
  contacts with compliant ALM, rejects ``False``, and ignores ``True``; omit the field.
