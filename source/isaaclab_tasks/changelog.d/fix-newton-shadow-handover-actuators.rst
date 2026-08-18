Fixed
^^^^^

* Fixed the Newton Shadow Hand handover configuration to avoid targeting
  distal ``J0`` joints that are absent from the current Newton asset.
* Fixed ``physics=isaacsim_physx`` for Shadow Hand handover to select the
  PhysX hand and object assets together with the PhysX scene.
* Fixed the G1 Newton agent preset to use its intended 5,000-iteration
  training schedule.
