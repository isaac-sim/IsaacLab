Changed
^^^^^^^

* Made observation copying automatic without task-level settings or producer annotations. Clipping
  and scaling established independent storage when needed; term and custom callback outputs were
  handled conservatively. Single-term groups avoided the concatenation copy, and dictionary and
  history outputs became independent snapshots.
* Created ``image_features`` normalization statistics once instead of on every inference call.
