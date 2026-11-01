* **Breaking:** Replay follows recorded START and RESET controls by default. Add
  ``--start_on_replay`` for recordings made without START events that should begin
  immediately.
* Changed replay to launch Kit only when the resolved task configuration requires it (for
  example Isaac Sim PhysX, ``--xr`` or a Kit visualizer) instead of always. Select a Kit
  physics preset to keep the previous runtime for a kitless task.
