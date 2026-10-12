* Added :attr:`~isaaclab.envs.ManagerBasedRLEnvCfg.capture_step`: :class:`~isaaclab.envs.ManagerBasedRLEnv` lays out
  its step as operations and replays the graphable ones from CUDA graphs, with eager islands for the rest, when the
  physics backend exposes its step operations (Newton). :attr:`~isaaclab.envs.ManagerBasedRLEnv.step_graph` shows the
  layout, and :meth:`~isaaclab.envs.ManagerBasedRLEnv.invalidate_step_graph` records the step again.
* Added :attr:`~isaaclab.managers.ManagerBase.supports_graph_capture`: a manager's step work is recorded unless a term
  function or class sets ``supports_graph_capture = False``.
* Added ``notify_update_replayed`` and ``reset_supports_graph_capture`` to assets, sensors and
  :class:`~isaaclab.scene.InteractiveScene`, :attr:`~isaaclab.scene.InteractiveScene.entities`, and an ``entities``
  argument to :meth:`~isaaclab.scene.InteractiveScene.reset`. Sensors declare ``supports_graph_capture``.
* :meth:`~isaaclab.managers.EventManager.apply` accepts the reset step count as a device tensor.
* Added :attr:`~isaaclab.envs.DirectRLEnvCfg.capture_step`: :class:`~isaaclab.envs.DirectRLEnv` captures its step the
  same way, with the default index-based reset as an eager island. Both workflows share
  :class:`~isaaclab.envs.utils.step_capture.EnvStepCapture`.
