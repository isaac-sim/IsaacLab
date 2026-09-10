Fixed
^^^^^

* Fixed a shutdown hang in ``random_agent``/``zero_agent`` and every ``train``/``play`` RL
  backend (``rsl_rl``, ``rl_games``, ``skrl``, ``sb3``) when interrupted with Ctrl+C while
  multiple GUI-backed visualizers were active (for example ``--visualizer newton,kit``).
  ``env.close()`` previously only ran on the happy path -- ``except KeyboardInterrupt: pass``
  silently swallowed the interrupt without closing the environment, so the process never
  reached the visualizer teardown in :meth:`~isaaclab.sim.SimulationContext.clear_instance`
  and had to be killed. ``env.close()`` is now registered on a :class:`contextlib.ExitStack`
  immediately once the environment is created, so it always runs, including for an interrupt
  during setup (``env.reset()``, wrapper/runner/checkpoint construction), not just one during
  the final training or playback loop.
  In ``random_agent``/``zero_agent`` and every ``play`` backend, a raw ``KeyboardInterrupt``
  could also be delivered asynchronously inside one of Kit's own internal callbacks (its
  viewport/render event loop) rather than in the script's own loop; those entrypoints now
  install a ``SIGINT`` handler (``isaaclab_rl.entrypoints.common.flag_only_sigint``) that only
  sets a flag, checked at a controlled point in the loop, instead of relying on where the
  exception happens to land. The flag-only handler stays installed through ``env.close()``
  itself, since restoring it beforehand let a second Ctrl+C during the multi-stage cleanup
  raise a raw ``KeyboardInterrupt`` that bypassed ``clear_instance``'s per-stage error
  isolation and could skip whatever visualizers had not closed yet. The ``train`` backends
  cannot use that handler -- their step loop runs inside a third-party runner's blocking call
  (``runner.learn()``/``runner.run()``), which needs a raw ``KeyboardInterrupt`` to interrupt
  it -- so they instead suppress the interrupt only once it has propagated past the
  ``ExitStack``, guaranteeing ``env.close()`` already ran.
