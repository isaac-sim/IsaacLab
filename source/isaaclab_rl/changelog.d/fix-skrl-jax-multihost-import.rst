Fixed
^^^^^

* Fixed :func:`~isaaclab_rl.skrl.SkrlVecEnvWrapper` failing to import the JAX wrapper on recent JAX
  versions by preloading the ``jax.experimental.multihost_utils`` submodule that skrl's distributed
  models reference without importing.
