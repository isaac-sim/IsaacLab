Changed
^^^^^^^

* Changed the Newton visualizer "no display found" notice to use ``logger.warning`` instead of ``print``.

Fixed
^^^^^

* Fixed the :class:`~isaaclab_visualizers.newton.NewtonVisualizerCfg` deprecation warning pointing at
  ``configclass.py`` instead of the code that constructed the config.
