Added
^^^^^

* Added unified ``train`` and ``play`` console-script entry points (``isaaclab.cli:train``
  and ``isaaclab.cli:play``) that dispatch to a library-specific implementation via
  ``--rl_library``. Supported libraries are ``rsl_rl``, ``rl_games``, ``skrl``, ``sb3``,
  and ``rlinf``.
* Added refactored per-library train/play scripts under
  ``scripts/reinforcement_learning/`` with a shared ``common.dispatch_library_entrypoint``
  helper, replacing the previous standalone per-library scripts.
* Added experimental ``uv run`` workflow allowing ``uv run train`` and ``uv run play``
  directly from the repository root without manual environment setup. See
  :ref:`uv-run-training` for usage.
