* Fixed lazily read OVRTX cameras rendering every camera of the renderer again for each read. A read now captures
  the other due cameras of its renderer that share its batch implementation in the same submission, so a step that
  reads N cameras renders once instead of N times. Due cameras that are not read are published with them.
