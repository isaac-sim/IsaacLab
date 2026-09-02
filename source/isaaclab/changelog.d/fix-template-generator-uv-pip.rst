Changed
^^^^^^^

* Changed generated projects to use the Newton backend without Isaac Sim by default. Use the ``isaacsim``, ``ov``,
  ``ovphysx``, or ``ovrtx`` uv extra when running a generated project that needs the corresponding optional backend.

Fixed
^^^^^

* Fixed the new project template generator in uv environments that do not include the ``pip`` module by declaring
  Jinja as an Isaac Lab dependency and using the existing Rich dependency for interactive prompts.
