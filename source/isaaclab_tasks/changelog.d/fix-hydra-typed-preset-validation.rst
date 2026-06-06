Added
^^^^^

* Added validation for the typed preset selectors ``physics=NAME`` and
  ``renderer=NAME`` during Hydra resolution. A typed selector is now enforced
  to resolve against a config of that type (a
  :class:`~isaaclab.physics.PhysicsCfg` / renderer config) at least once;
  selecting one on a task that only exposes the name as an unrelated preset
  (e.g. a scalar or sensor variant) raises a descriptive :class:`ValueError`
  instead of silently leaving the backend unchanged. The free-form
  ``presets=NAME`` broadcast is trusted and not enforced.

Changed
^^^^^^^

* **Breaking:** Removed ``isaaclab_tasks.utils.fold_preset_tokens``.
  :func:`~isaaclab_tasks.utils.preset_cli.setup_preset_cli` now returns the
  ``physics=`` / ``renderer=`` / ``presets=`` tokens verbatim, and
  :func:`~isaaclab_tasks.utils.hydra.register_task` parses them directly.
  Scripts assign the remainder to ``sys.argv`` unchanged (drop the
  ``fold_preset_tokens(...)`` wrapper).
