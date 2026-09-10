Changed
^^^^^^^

* **Breaking:** Changed the pinned standalone OpenUSD provider from ``usd-exchange`` 2.3.0 to
  3.0.0. In kit-less installs ``pxr`` is now OpenUSD **26.08** instead of 25.05. Environments
  that pin Isaac Lab against OpenUSD 25.05 outside Kit must re-resolve, and any code compiled
  or generated against the 25.05 ``pxr`` ABI must be rebuilt against 26.08. Kit-backed runs are
  unaffected: Kit serves its own OpenUSD from its extension roots. ``isaacsim-asset-isolated``
  and ``isaacsim-robot-schema`` pin ``usd-exchange`` exactly and are forced to 3.0.0 through
  ``[tool.uv] override-dependencies``.
* **Known limitation:** ``UrdfConverterCfg.merge_mesh`` and ``MjcfConverterCfg.merge_mesh``
  silently have no effect in kit-less installs, leaving the model unmerged, because
  ``usd-optimize`` cannot read OpenUSD 26.08 stages. Run the conversion under Kit until
  ``usd-optimize`` ships a 26.08 build.
