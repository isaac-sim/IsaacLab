Fixed
^^^^^

* Fixed preset-based ``--agent`` auto-selection being skipped for every entrypoint that registers
  ``--agent`` with a non-``None`` default (``rsl_rl``, ``rl_games`` and ``sb3``). The selection guard
  could not tell a default-supplied value from a user-typed one, so ``presets=resnet18`` and
  ``presets=theia_tiny`` on ``Isaac-Cartpole-Camera`` kept the raw-camera entry point and the runner
  failed to construct. An explicitly typed ``--agent`` still wins over auto-selection.
