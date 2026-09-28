Fixed
^^^^^

* Fixed failed retrieval leaving an empty prim, dependencies resolving outside the destination stage's resolver
  context, and Windows drive paths being treated as URLs.

* Removed stale resolved-tree reuse so newly available dependencies and changed search-path precedence were
  respected. Downloads remained cached; failed metadata queries became retryable and forced retrieval refreshed
  remote revision metadata.

* Preserved source anchors for unresolved USD references, MDL modules, and UDIM textures instead of rejecting
  unselected variants or redirecting materials to incomplete local mirrors. Material dependency loading remained
  with the renderer, including Kit's built-in MDL module names. Standalone ``retrieve_file_path`` callers using
  search paths must bind their resolver context. ``add_usd_reference`` and ``create_prim`` bind the destination
  stage's context automatically.
