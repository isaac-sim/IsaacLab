* **Breaking:** Removed ``isaaclab_experimental.managers.SceneEntityCfg``. Use :class:`isaaclab.managers.SceneEntityCfg`;
  the Warp managers fill its Warp selection fields through :meth:`~isaaclab.managers.SceneEntityCfg.resolve_warp`.
* **Breaking:** Removed ``isaaclab_experimental.utils.CapturedStage``, ``captured``, ``eager`` and
  ``reset_captured_stages``, and ``WarpCapturable``, ``is_warp_capturable`` and ``warp_capturable`` from
  ``isaaclab_experimental.utils.warp``. Import them from :mod:`isaaclab.utils.warp.capture`, and use
  ``@WarpCapturable(False)`` in place of ``@warp_capturable(False)``.
