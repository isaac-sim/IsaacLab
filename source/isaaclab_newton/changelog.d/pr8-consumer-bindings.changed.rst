* Reused shared body poses for legacy multi-mesh ray-cast targets, removing duplicate
  target sites and per-update temporary pose arrays.
* Refreshed shared body poses before updating ray origins so manual pose writes were
  visible to the sensor and its targets in the same update.
