Added
^^^^^

* Extended the visualizer "Reset Episode" button to work in :class:`~isaaclab.envs.DirectRLEnv`
  and :class:`~isaaclab.envs.DirectMARLEnv` (train and play). Previously only
  :class:`~isaaclab.envs.ManagerBasedRLEnv` consumed the reset request from the visualizer.

Fixed
^^^^^

* Fixed duplicate "Training Metrics" live-plot checkbox in the Kit visualizer UI. The checkbox
  was created by both :class:`~isaaclab.envs.ui.BaseEnvWindow` and
  :class:`~isaaclab.envs.ui.ManagerBasedRLEnvWindow`; it is now created only by the base window.
* Fixed :class:`~isaaclab.ui.widgets.ManagerLiveVisualizer` and
  :class:`~isaaclab.ui.widgets.DirectScalarLiveVisualizer` raising
  ``UnboundLocalError: cannot access local variable 'omni'`` when a live-plot
  checkbox was toggled a second time. The ``omni.kit.app`` import was inside a
  conditional branch that was skipped on re-entry, leaving ``omni`` unbound when
  ``omni.ui`` was used later in the same function.
