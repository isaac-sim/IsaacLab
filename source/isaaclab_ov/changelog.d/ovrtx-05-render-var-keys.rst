Added
^^^^^

* Added compatibility with the OVRTX 0.5 ``frame.render_vars`` API, which keys render vars by the
  authored RenderVar prim path (for example ``/Render/Vars/LdrColor``) instead of the source name.
  The key form is resolved from the installed ``ovrtx`` version when
  :mod:`isaaclab_ov.renderers.ovrtx_compat` is imported; OVRTX 0.4 keeps source-name keys and the
  public extras stay pinned to ``ovrtx==0.4.1.364340``.
