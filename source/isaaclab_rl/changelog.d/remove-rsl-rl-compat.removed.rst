* **Breaking:** Removed the legacy RSL-RL policy configuration classes, deprecated model fields,
  and automatic configuration migration. Define ``actor`` and ``critic`` or ``student`` and
  ``teacher`` model configurations directly, and use ``distribution_cfg`` for stochastic models.
* **Breaking:** Removed ``export_policy_as_jit`` and ``export_policy_as_onnx`` from
  ``isaaclab_rl.rsl_rl``. Use the RSL-RL runner methods ``export_policy_to_jit`` and
  ``export_policy_to_onnx`` instead.
* Removed obsolete runtime RSL-RL minimum-version checks; the project dependency provides
  the supported version.
