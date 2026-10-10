* Fixed state-dependent actions and explicit actuators being excluded from backend-owned decimation. Environments
  bind their ordinary action and scene writers to the physics manager; graph-safe work executes inside capture.
* Fixed manager instances sharing callback IDs, callbacks, views, configuration, and simulation time. Resources
  now belong to their manager, including deterministic STOP dispatch and cleanup after callback failures.
