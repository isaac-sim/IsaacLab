Added
^^^^^

* Added four consolidated Cartpole perception tasks that subsume 35
  per-variant task IDs via the typed preset CLI (#5587):
  ``Isaac-Cartpole-Camera-Direct-v0``, ``Isaac-Cartpole-Camera-v0``,
  ``Isaac-Cartpole-Showcase-Direct-v0``, and
  ``Isaac-Cartpole-Camera-Showcase-Direct-v0``. Variant (data type,
  observation pipeline, gym-space shape) selected at runtime via
  ``presets=<name>``; agent yaml selected via
  ``--agent=<entry_point_name>`` for the manager perception feature
  policies and all non-default showcase shapes.
* Added :func:`isaaclab_tasks.utils.deprecated_task_alias` -- factory
  that wraps a retired gym task ID with a :class:`DeprecationWarning`
  naming the consolidated task plus its equivalent CLI tokens, and
  delegates cfg construction to a caller-supplied ``cfg_factory``.

Deprecated
^^^^^^^^^^

* Deprecated 35 per-variant Cartpole task IDs (7 Direct-backend camera,
  4 manager-based camera, 15 proprioceptive showcase, 9 camera-based
  showcase) in favor of the four consolidated tasks above. Each retired
  ID still loads and emits a :class:`DeprecationWarning` naming the
  consolidated task and the equivalent ``presets=<name>`` (plus
  ``--agent=<entry_point_name>`` where required) invocation. The shims
  return the historical per-variant cfg subclass verbatim so retired
  IDs stay bit-for-bit identical to their pre-deprecation behavior;
  only the deprecation warning is layered on top. The historical
  subclasses (e.g. ``CartpoleRGBCameraEnvCfg``,
  ``CartpoleAlbedoCameraEnvCfg``, ``BoxBoxEnvCfg``, ...) are kept for
  one release alongside the consolidated cfgs and will be removed
  together with the retired task IDs. Full migration table is in the
  PR description.
