* **Breaking:** Removed ``isaaclab_experimental.managers.SceneEntityCfg``. Use :class:`isaaclab.managers.SceneEntityCfg`;
  the Warp managers fill its Warp selection fields through :meth:`~isaaclab.managers.SceneEntityCfg.resolve_warp`.
* **Breaking:** Removed ``isaaclab_experimental.utils.CapturedStage``, ``captured``, ``eager`` and
  ``reset_captured_stages``. Import them from :mod:`isaaclab.utils.warp.capture`.
* **Breaking:** Removed ``isaaclab_experimental.utils.warp.warp_capturable``. Use ``@WarpCapturable(False)`` in its
  place.
