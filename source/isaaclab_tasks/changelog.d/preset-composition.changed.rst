* Simplified task preset traversal and CLI help, preserving selection validation and deprecated-name warnings.
* **Breaking:** Consolidated task preset parsing and categories in ``isaaclab_tasks.utils.hydra``.
  Replace direct imports from ``isaaclab_tasks.utils.preset_cli`` and ``isaaclab_tasks.utils.preset_target``
  with imports from ``isaaclab_tasks.utils.hydra``; ``isaaclab_tasks.utils.setup_preset_cli`` stayed unchanged.
