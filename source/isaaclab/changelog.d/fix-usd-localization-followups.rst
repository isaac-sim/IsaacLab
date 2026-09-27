Fixed
^^^^^

* Fixed repeated USD localization creating redundant working copies, failed retrieval leaving an empty prim,
  dependencies resolving outside the destination stage's resolver context, and Windows drive paths being
  treated as URLs.

* Preserved source anchors for unresolved USD references, MDL modules, and UDIM textures instead of rejecting
  unselected variants or redirecting materials to incomplete local mirrors. Material dependency loading remained
  with the renderer. Standalone ``retrieve_file_path`` callers using search paths must bind their resolver context;
  ``add_usd_reference`` and ``create_prim`` bind the destination stage's context automatically.
