Changed
^^^^^^^

* Changed ``Metrics/success_rate`` for the Shadow Hand handover task to report whether the object is
  at the goal when the episode ends. It previously latched as soon as the object first came within
  the success distance, so an object swung through the goal scored the same as one left resting
  there. Both the manager-based and direct environments were updated together. Reported success
  rates are lower than before for the same policy, and are not comparable with values recorded
  under the previous definition; re-evaluate any checkpoint whose success rate is being compared
  across this change.
