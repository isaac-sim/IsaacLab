* Simplified keyboard sources to listener registration and event forwarding; native window hooks
  were initialized and closed by visualizers instead of changing with listener counts.
* Centralized the optional keyboard source and teardown in the base visualizer and removed
  redundant subscription state.
