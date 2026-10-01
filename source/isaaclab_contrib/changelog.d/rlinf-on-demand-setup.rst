Added
^^^^^

* Added GR00T N1.7 support using RLinf's native model loader and checkpoint-owned processor metadata.
* Added optional ordered ``action_mapping.keys`` to concatenate policy outputs without task-specific converters.

Fixed
^^^^^

* Fixed stale action chunks driving freshly reset environments. Tasks enabling
  ``hold_pose_on_midchunk_reset`` held their returned joint positions until the next chunk.
* Fixed the RLinf extension accepting a direct ``full_weights.pt`` path as well as its checkpoint directory.
* Fixed integration with RLinf 0.3's generation-specific model and action-converter modules.
