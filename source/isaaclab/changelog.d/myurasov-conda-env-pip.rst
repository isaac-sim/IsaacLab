Fixed
^^^^^

* Fixed ``./isaaclab.sh -p -m pip ...`` failing with ``No module named pip``
  in the conda env created from ``environment.yml`` on Linux aarch64
  (e.g. DGX Spark / GB10). The conda-forge solver was not pulling
  ``pip`` in transitively on aarch64, so the resulting ``env_isaaclab``
  had no pip. ``environment.yml`` now lists ``pip`` explicitly so it
  is seeded on every platform.
