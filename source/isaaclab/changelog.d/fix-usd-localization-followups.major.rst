Fixed
^^^^^

* Fixed repeated USD localization creating redundant working copies, failed retrieval leaving an empty prim,
  dependencies resolving outside the destination stage's resolver context, and Windows drive paths being
  treated as URLs.

Changed
^^^^^^^

* **Breaking:** Made USD preparation fail on missing file dependencies instead of returning partial assets.
  Make required files available before spawning, then retry the original asset path. Standalone callers of
  ``retrieve_file_path`` using search paths must bind the intended USD resolver context; ``add_usd_reference``
  and ``create_prim`` bind the destination stage's context automatically. Renderer module identifiers and
  USDZ contents retained their native loader ownership.
