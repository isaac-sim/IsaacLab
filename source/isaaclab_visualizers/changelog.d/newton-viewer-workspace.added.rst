* Added support for Newton's experimental viewer workspace, including native simulation transport,
  isolated geometry-history playback, dockable inspection and plots, and Isaac Lab window branding.
* Added :meth:`~isaaclab_visualizers.newton.NewtonGLVisualizer.window_session` to retain native
  windows and workspace layouts across independently owned simulations.
* Added an Isaac Lab green viewer theme, CLI robot logo, and animated loading splash when
  switching packaged examples in the retained window.
* Added configurable workspace history recording, disabled by default during RL training
  to avoid device-to-host snapshot transfers until recording is enabled.
