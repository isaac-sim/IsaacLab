Added
^^^^^

* Added :mod:`isaaclab_experimental.envs.frontend`, a small framework that
  selects the runtime backend for IsaacLab tasks. A :class:`Frontend` takes
  a stable env cfg + task id, runs a pluggable :class:`CompatRule` pipeline
  against the (cfg, task, frontend) triple, and constructs the env on the
  matching runtime. New runtimes plug in by subclassing :class:`Frontend`
  and calling :func:`register_frontend`; new compatibility checks plug in
  by subclassing :class:`CompatRule` and listing it in a frontend's
  :attr:`Frontend.rules`.

  The framework ships two built-in frontends:

  * :class:`TorchFrontend` (``--frontend=torch``, default) — passes through
    to :func:`gym.make`. Emits a warning if the task is registered under
    the warp runtime.
  * :class:`WarpFrontend` (``--frontend=warp``) — adapts a manager-based
    stable cfg onto :class:`ManagerBasedRLEnvWarp` via the rule pipeline,
    or dispatches a direct task to its registered warp env class.

  The default warp rules are:

  * :class:`CheckPhysicsIsNewton` — blocking check; PhysX physics with the
    warp runtime is a hard incompatibility (``isaaclab_physx`` classes
    depend on ``omni.physics.tensors``, which the warp runtime does not
    initialise).
  * :class:`ResolvePhysicsPreset` — collapses ``PresetCfg`` to its
    ``newton`` field for programmatic callers (Hydra's preset resolution
    handles the CLI case).
  * :class:`DropUnsupportedSensors` — drops scene sensors warp can't run.
  * :class:`PromoteSceneEntityCfg` — in-place class promotion for warp
    cached fields, with an :func:`issubclass` assertion to fail loudly if
    the warp class ever stops subclassing the stable one.
  * :class:`SwapMdpFunctions` — name-based ``term.func`` replacement
    against the warp ``mdp`` modules; rejects stable re-exports by
    inspecting ``__module__``.
  * :class:`SwapActionClassType` — strict swap of action ``class_type``
    to the warp twin; missing twin is :attr:`Severity.BLOCKING`.
  * :class:`VerifyDirectIsWarp` — for direct cfgs, blocks if the
    registered entry-point isn't a warp class.

  All rules emit :class:`Issue` (incompatibility) and / or :class:`Change`
  (transformation applied) records into a :class:`Report` accessible on
  ``env.unwrapped.frontend_report`` after :meth:`Frontend.build`.

  :class:`TaskResolver` centralises ``gym.spec`` introspection and
  classifies a task into a :class:`TaskMeta` (workflow, registered
  runtime, cfg class). Rules read from :class:`TaskMeta` rather than
  poking gym directly.

* Added a ``--frontend={torch,warp}`` flag to ``rsl_rl/train.py``. The
  script delegates Hydra-arg pre-processing (``presets=newton`` injection,
  conflicting-preset warnings) and env construction to the selected
  frontend via :func:`get_frontend`; ``render_mode`` is forwarded so
  ``--video`` keeps working under the warp frontend.

Fixed
^^^^^

* Fixed a regression in :class:`~isaaclab_experimental.envs.ManagerBasedRLEnvWarp`
  introduced when the ``SimulationContext.get_setting`` API was reshaped:
  the warp env now mirrors the stable env and probes
  :meth:`~isaaclab.sim.SimulationContext.has_active_visualizers` instead of
  splitting a string setting that no longer exists.
