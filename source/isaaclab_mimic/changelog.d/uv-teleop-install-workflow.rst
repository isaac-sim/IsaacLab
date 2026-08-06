Fixed
^^^^^

* Fixed ``ModuleNotFoundError: No module named 'ipywidgets'`` when running dataset generation
  from an environment without the ``mimic`` extra. ``isaaclab_mimic.datagen.utils`` imported
  ``ipywidgets`` and ``IPython`` at module scope even though only its interactive notebook
  helpers use them, so importing the module for its path helpers pulled in dependencies the
  generation path never needs.
