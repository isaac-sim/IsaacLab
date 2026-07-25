Added
^^^^^

* Added an RSL-RL training configuration and success metrics to the Shadow handover
  Direct task.
* Added OVPhysX physics presets to the handover and camera Direct environments.

Changed
^^^^^^^

* Changed the default physics backend of the Shadow handover Direct task from PhysX to
  Newton (MJWarp). Pass ``physics=physx`` for the previous backend.

Fixed
^^^^^

* Fixed handover construction on Newton, which raised ``No joints found for actuator
  group``.
* Fixed the Shadow hand root orientation on Newton, which left both palms rotated
  90 degrees.
* Fixed the handover goal orientation, which was initialized to a 180-degree rotation
  instead of identity.
