Changed
^^^^^^^

* **Breaking:** Changed environments whose default physics preset was automatic
  ``physx`` to use concrete ``isaacsim_physx``. Environments with explicit
  backend defaults, including Newton, remain unchanged. Select
  ``physics=physx`` to retain automatic PhysX-family resolution between Isaac
  Sim PhysX and OvPhysX.
