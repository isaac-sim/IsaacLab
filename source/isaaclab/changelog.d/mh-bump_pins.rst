Added
^^^^^

* Added an ``ov`` install extra that pulls both Omniverse backends (``ovphysx`` and ``ovrtx``);
  the per-backend ``ovphysx`` / ``ovrtx`` extras still select one.

Changed
^^^^^^^

* Changed the Isaac Sim pin to ``6.0.1.0`` and the Newton commit pin to the matching upstream
  revision. Reinstall the environment (e.g. ``uv sync``) after pulling this change.
* Changed the dependency environment markers to explicit ``platform_machine == '<arch>'``
  comparisons and restricted the uv resolution universe to the supported platforms
  (Linux x86_64, Linux aarch64, Windows AMD64).
