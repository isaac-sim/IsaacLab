Fixed
^^^^^

* Fixed :func:`~isaaclab_tasks.utils.parse_env_cfg` silently misinterpreting a bare ``overrides`` string as a
  sequence of single-character overrides (a plain string is itself a ``Sequence[str]``). Passing a bare string now
  raises a clear ``TypeError`` instructing the caller to wrap it in a list or tuple.
* Extended the Hydra-style ``overrides`` forwarding added for the OVD Recorder fix to every standalone script that
  calls :func:`~isaaclab_tasks.utils.parse_env_cfg`, so ``physics=``/``renderer=``/``presets=`` selectors work
  consistently across all of them, not just the ``run_cartpole_rl_env.py`` tutorial.
