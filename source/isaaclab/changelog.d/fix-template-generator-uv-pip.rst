Fixed
^^^^^

* Fixed the new project template generator in uv environments that do not include the ``pip`` module by declaring
  Jinja as an Isaac Lab dependency and using the existing Rich dependency for interactive prompts.
