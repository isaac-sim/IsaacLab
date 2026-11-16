* **Breaking:** Replaced Lift's ``conditional_reset`` parameter ``success_monitor`` with
  ``reset_sampler``. Configure the shared ``isaaclab.envs.reset.ResetStateSamplerCfg``;
  use ``history_length`` for outcome history and ``scores`` for the selection policy.
* **Breaking:** Migrated NIST to ``ResetStateSamplerCfg`` and removed its legacy sampling
  classes and import aliases. Replace ``SamplerCfg(strategies=...)`` with
  ``ResetStateSamplerCfg(scores=...)`` using core ``TargetSuccessScoreCfg`` or
  ``UniformScoreCfg``. Set ``history_length`` instead of passing ``success_monitor_cfg``.
