Changed
^^^^^^^

* Bumped the ``rsl-rl-lib`` dependency to ``5.5.1``.
* Removed the redundant direct ``onnxscript`` dependency, which is provided by ``rsl-rl-lib``.
* Removed obsolete runtime RSL-RL minimum-version checks; the exact project dependency now
  provides the supported version.

Removed
^^^^^^^

* **Breaking:** Removed the legacy RSL-RL policy configuration classes, deprecated model fields,
  and automatic configuration migration. Define ``actor`` and ``critic`` or ``student`` and
  ``teacher`` model configurations directly, and use ``distribution_cfg`` for stochastic models.
* **Breaking:** Removed ``export_policy_as_jit`` and ``export_policy_as_onnx`` from
  ``isaaclab_rl.rsl_rl``. Use the RSL-RL runner methods ``export_policy_to_jit`` and
  ``export_policy_to_onnx`` instead.
