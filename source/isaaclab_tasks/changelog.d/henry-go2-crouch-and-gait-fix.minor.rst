Changed
^^^^^^^

* Changed ``Isaac-Velocity-Flat-UnitreeGo2`` and ``Isaac-Velocity-Rough-UnitreeGo2`` to add a
  ``base_height_l2`` reward term at weight ``-30.0`` (rough, height-scanner adjusted) and disabled
  on flat via ``sensor_cfg=None``. Without it, Go2's leg colliders let a policy rest its weight on
  the legs and collect the full episode-length alive bonus from a permanent crouch, since torso
  contact never fires and flat-orientation reward can't see it.
* Changed ``action_rate_l2`` on both tasks from the shared default of ``-0.01`` to ``-0.005``. The
  stronger penalty locked one hind foot into a low-amplitude, dragging gait on flat terrain with
  the Newton backend. Policies trained on these tasks will differ from ones trained before this
  change.
