Fixed
^^^^^

* Fixed a shutdown hang in ``random_agent``/``zero_agent`` and every ``train``/``play`` RL
  backend (``rsl_rl``, ``rl_games``, ``skrl``, ``sb3``) when interrupted with Ctrl+C while
  multiple GUI-backed visualizers were active (for example ``--visualizer newton,kit``).
  ``env.close()`` previously only ran on the happy path -- ``except KeyboardInterrupt: pass``
  silently swallowed the interrupt without closing the environment, so the process never
  reached the visualizer teardown in :meth:`~isaaclab.sim.SimulationContext.clear_instance`
  and had to be killed. ``env.close()`` now runs from a ``finally`` block on every exit path.
  In ``random_agent``/``zero_agent``, a raw ``KeyboardInterrupt`` could also be delivered
  asynchronously inside one of Kit's own internal callbacks (its viewport/render event loop)
  rather than in the script's own loop; that entrypoint now installs a ``SIGINT`` handler that
  only sets a flag, checked at a controlled point in the loop, instead of relying on where the
  exception happens to land. The flag-only handler stays installed through ``env.close()``
  itself, since restoring it beforehand let a second Ctrl+C during the multi-stage cleanup
  raise a raw ``KeyboardInterrupt`` that bypassed ``clear_instance``'s per-stage error
  isolation and could skip whatever visualizers had not closed yet. The protected scope also
  now starts immediately after each environment is created, covering ``env.reset()`` and
  wrapper/runner/checkpoint setup, not just the final training or playback loop -- an
  interrupt during that setup work could bypass ``env.close()`` the same way.
