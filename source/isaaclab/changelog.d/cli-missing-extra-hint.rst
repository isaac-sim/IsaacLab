Added
^^^^^

* Added :func:`~isaaclab.utils.extras.missing_extra_hint`, which turns a missing optional
  dependency into a message naming the extra that provides it and the command to install it.
  ``isaaclab`` commands now report ``rl_games is not installed. It is provided by the
  'rl-games' extra`` instead of a bare ``ModuleNotFoundError``. The suggested command matches
  the environment: ``uv run --extra ...`` under uv, ``pip install "isaaclab-dev[...]"`` otherwise.
