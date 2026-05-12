Fixed
^^^^^

* Fixed ``apps/isaaclab.python.kit`` referencing three Isaac Sim extensions
  that don't exist in publicly-released ``isaacsim 6.0.x``:
  ``isaacsim.core.experimental.primdata``,
  ``isaacsim.robot.wheeled_robots.nodes``, and
  ``isaacsim.sensors.experimental.rtx``. The references were added in #5293
  as part of the deprecated-extension migration, but the renamed targets
  only exist in unreleased Isaac Sim builds. Kit's resolver falls back to
  remote registry sync for every missing dependency, which silently burns
  ~55s locally and 1000s+ in CI (the registry endpoints are slow /
  unreachable from runners), triggering recent "isaaclab (core)" per-test
  timeouts on ``test_non_headless_launch.py``, ``test_outdated_sensor.py``,
  and ``test_color_randomization.py``. Marking the three as
  ``{ optional = true }`` makes Kit's resolver tolerate their absence while
  still picking them up automatically once they ship in a future Isaac Sim
  release.
* Fixed :func:`~isaaclab.envs.mdp.events.randomize_visual_color` and three
  sibling event terms crashing with
  ``AttributeError: 'NoneType' object has no attribute 'split'`` when
  ``omni.replicator.core`` is loaded as a namespace package by Kit's
  extension manager (which leaves ``rep.__file__ = None``). Replaced the
  fragile ``rep.__file__.split("/")[-5][21:]`` version extraction with a
  dedicated helper ``_get_replicator_version`` that falls back to
  ``rep.__path__[0]`` -- always populated -- and uses ``re.search`` rather
  than positional slicing, so it survives both packaging modes.
