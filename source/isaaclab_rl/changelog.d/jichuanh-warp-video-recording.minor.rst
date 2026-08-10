Changed
^^^^^^^

* Removed the restriction that ``--video`` requires ``--frontend torch``. Video recording now
  works on the Warp frontend.
* Changed the visualizer auto-created for ``--video`` when no ``--viz`` is given: the Warp
  frontend now gets a headless Newton GL visualizer instead of a headless Kit one, since the
  Warp runtime does not initialise Kit. The Torch frontend is unchanged.
