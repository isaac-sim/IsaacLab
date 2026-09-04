Fixed
^^^^^

* Fixed :meth:`~isaaclab_visualizers.newton.NewtonGLVisualizer.render_rgb_array` omitting
  visualization markers, so videos recorded with ``--viz newton_gl`` showed the scene without
  its goal poses, command arrows, and other debug markers visible in the interactive viewer.
