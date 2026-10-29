* Fixed lazily read cameras of a renderer that renders all cameras per submission rendering every camera again for
  each read. A read now captures the other due cameras of its renderer that share its batch implementation with it,
  so a step that reads N cameras renders once instead of N times, and every camera gets one rendered frame per
  capture regardless of the read order.
