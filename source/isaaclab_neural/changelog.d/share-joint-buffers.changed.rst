* Shared the world-model manager's joint-state and target tensors with its virtual articulation, removing per-step
  host copies. Joint inputs are converted to the existing NumPy model interface only at chunk boundaries; reset
  events and the configured default joint pose retain their behavior.
