* Kept live gravity updates separate for each physics scene, so editing another scene
  does not change the gravity reported by the manager.
* Allowed compatible CPU and CUDA backends to be created after another backend starts
  stepping without reapplying unchanged OvPhysX startup settings, including equivalent
  cache paths. A conflicting startup setting now raises an error naming the fixed value.
* Rejected premature scene-data reads without consuming their deferred setup, allowing
  publication to initialize after warmup or the first simulation step.
