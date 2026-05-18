Fixed
^^^^^

* Fixed slow OVPhysX warmup at large env counts. The previous
  :meth:`~isaaclab_ovphysx.physics.OvPhysxManager._warmup_and_load`
  flattened the entire post-clone USD stage to disk before stripping
  ``env_1..N`` from the resulting file — a ~31 s flatten at 4096 envs
  on Anymal-D Rough. The manager now snapshots the live stage from
  inside :func:`~isaaclab_ovphysx.cloner.ovphysx_replicate` (which runs
  before :func:`cloner.usd_replicate` inflates the stage), and
  :meth:`_warmup_and_load` consumes the snapshot directly. The old
  export-and-strip path is preserved as a fallback for callers that do
  not go through :meth:`isaaclab.scene.InteractiveScene.clone_environments`.
