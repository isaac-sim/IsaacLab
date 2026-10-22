* **Breaking:** Replaced the legacy rough G1 configuration with the teacher configuration under
  ``Isaac-Velocity-Rough-G1``. Changed callers of ``Isaac-Velocity-Rough-G1-Teacher`` to use that task ID;
  callers of ``G1RoughEnvCfg`` and ``G1RoughPPORunnerCfg`` must use ``G1RoughTeacherEnvCfg`` and
  ``G1RoughTeacherPPORunnerCfg``. Preserved the teacher, depth-student and flat task settings.
