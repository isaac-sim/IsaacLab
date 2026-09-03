Changed
^^^^^^^

* **Breaking:** Updated ``SO101_CFG`` to use the SysID-capable asset and resolve actuator gains, friction, armature,
  and limits from its default Newton MJWarp USD variant. The USD-authored actuator group is now named ``usd``. The
  config also uses the workshop operational joint pose, inherits root fixation from the USD, disables
  self-collisions, enables contact sensors, and applies a 0.98 soft joint-limit factor. Tasks that require the
  previous simulation gains should migrate to ``SO101_HIGH_PD_CFG``, which retains the prior high-PD actuator
  behavior.
