Added
^^^^^

* Added :data:`~isaaclab.cloner.REPLICATION_QUEUE` and the free function
  :func:`~isaaclab.cloner.replicate`, the explicit registry-and-drain pair
  that backends now hook into for replication.
* Added :meth:`~isaaclab.cloner.ClonePlan.from_env_0` for direct envs that
  clone a single env-0 prototype across every env.
* Added :attr:`~isaaclab.cloner.CloneCfg.clone_regex` as the single source
  of truth for the env-namespace convention (default ``"/World/envs/env_.*"``).

Fixed
^^^^^

* Fixed :data:`~isaaclab.cloner.REPLICATION_QUEUE` leaking stale entries
  when a backend or asset construction raised mid-session.

Changed
^^^^^^^

* **Breaking:** Rewrote :class:`~isaaclab.cloner.ReplicateSession` as a thin
  context manager around :func:`~isaaclab.cloner.make_clone_plan` and
  :func:`~isaaclab.cloner.replicate`. The no-arg form and the cached
  ``plan`` / ``cfg_rows`` / ``replicate_on_exit`` fields are gone. Direct
  envs migrate to ``cloner.replicate(cloner.ClonePlan.from_env_0(...))``.
* **Breaking:** Changed :func:`~isaaclab.cloner.make_clone_plan` to take
  ``cfgs`` and absorb the cfg-driven planning logic previously inside
  :class:`~isaaclab.scene.InteractiveScene`, returning a self-contained
  :class:`~isaaclab.cloner.ClonePlan`.
* **Breaking:** :func:`~isaaclab.cloner.replicate` and
  :class:`~isaaclab.cloner.ReplicateSession` now require an explicit
  ``stage=`` keyword; the :class:`~isaaclab.cloner.ClonePlan` is
  stage-agnostic.
* Changed :attr:`~isaaclab.scene.InteractiveScene.env_origins` to read from
  the published :class:`~isaaclab.cloner.ClonePlan`, making the plan the
  single source of truth for env placement.

Removed
^^^^^^^

* **Breaking:** Removed ``isaaclab.cloner.replicate_session_defaults`` and
  ``isaaclab.cloner.replicate_session``. Use
  :data:`~isaaclab.cloner.REPLICATION_QUEUE` and
  :func:`~isaaclab.cloner.replicate` instead.
* **Breaking:** Removed :meth:`InteractiveScene.clone_environments`; direct
  envs should use ``cloner.replicate(cloner.ClonePlan.from_env_0(...))``.
* **Breaking:** Removed :attr:`InteractiveScene.env_ns` and
  :attr:`InteractiveScene.env_regex_ns`; read
  :attr:`~isaaclab.cloner.CloneCfg.clone_regex` instead.
