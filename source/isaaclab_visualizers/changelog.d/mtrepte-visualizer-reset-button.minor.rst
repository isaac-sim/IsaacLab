Added
^^^^^

* Added interactive sidebar controls to :class:`~isaaclab_visualizers.viser.ViserVisualizer`:
  **Pause Rendering** (freezes the 3D view without stopping physics), **Pause Simulation**
  (pauses the training/rollout loop via :meth:`~isaaclab.visualizers.BaseVisualizer.is_training_paused`),
  and **Reset Episode** (signals an episode reset via
  :meth:`~isaaclab.visualizers.BaseVisualizer.consume_reset_request`).
  Both pause buttons update their label and turn orange when active.
