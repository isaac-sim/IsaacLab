Added
^^^^^

* Added warp-first Direct variants of the Shadow cube reorientation and handover
  tasks (``Isaac-Reorient-Cube-Shadow-Direct-Warp-v0``,
  ``Isaac-Reorient-Cube-Shadow-OpenAI-FF-Direct-Warp-v0``,
  ``Isaac-Reorient-Cube-Shadow-OpenAI-LSTM-Direct-Warp-v0``, and
  ``Isaac-Shadow-Handover-Direct-Warp-v0``). The environments compute rewards,
  observations, and resets in Warp kernels and reuse the stable configurations
  and agents from :mod:`isaaclab_tasks.core`.
