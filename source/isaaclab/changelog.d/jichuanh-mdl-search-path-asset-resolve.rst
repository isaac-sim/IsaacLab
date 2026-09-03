Fixed
^^^^^

* Fixed :func:`~isaaclab.sim.utils.resolve_paths` rewriting search-path asset identifiers, such as
  the MDL module ``OmniPBR.mdl``, into paths relative to the process working directory. Assets
  converted with ``--make-instanceable`` referenced a non-existent MDL module, so the renderer
  logged ``MDLC comp error: C120 could not find module`` and left their materials unresolved.
