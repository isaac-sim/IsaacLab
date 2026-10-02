* Fixed stale action chunks driving freshly reset environments. Tasks enabling
  ``hold_pose_on_midchunk_reset`` held their returned joint positions until the next chunk.
* Fixed integration with RLinf 0.3's generation-specific model and action-converter modules.
