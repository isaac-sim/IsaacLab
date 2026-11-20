* Shared the world-model manager's joint-state and target tensors with its virtual articulation, removing per-step
  host copies while retaining explicit reset events and the configured default joint pose.
