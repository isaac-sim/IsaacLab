* Fixed demonstration recording targets changing within an episode by deferring
  command resampling until the next recording attempt resets the environment.
* Fixed HDF5 and Isaac Capture replay startup for tasks whose rewards referenced
  success terminations, while retaining manual success validation.
* Disabled unused training rewards and curricula during Robomimic policy evaluation
  so manually checked success conditions could be removed from automatic terminations.
