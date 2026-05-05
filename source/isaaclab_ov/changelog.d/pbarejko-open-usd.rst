Fixed
^^^^^

* Fixed ``AttributeError: 'Renderer' object has no attribute 'add_usd'`` in
  :class:`~isaaclab_ov.renderers.OVRTXRenderer` when using ``ovrtx`` 0.3.0 or
  newer. The renderer now calls :meth:`ovrtx.Renderer.open_usd` on 0.3.0+ and
  falls back to ``Renderer.add_usd`` on older versions.
