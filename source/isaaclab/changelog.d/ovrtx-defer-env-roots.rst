Changed
^^^^^^^

* Changed replicated stage preparation to retain only clone-plan prototypes until
  :attr:`~isaaclab.physics.PhysicsEvent.MODEL_INIT`. Custom :class:`~isaaclab.cloner.ClonePlan`
  instances that expect a backend to materialize environment roots must set
  :attr:`~isaaclab.cloner.ClonePlan.env_template`; the provided constructors set it automatically.
  Renderer implementations that require destination environments must expand the clone plan in a
  private stage or defer that work until replication completes.

Fixed
^^^^^

* Fixed replicated-scene startup scaling by deferring destination environment roots until backend
  initialization, so asset spawners author each prototype only once.
