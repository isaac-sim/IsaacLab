Changed
^^^^^^^

* **Breaking:** Changed the pinned standalone OpenUSD provider from ``usd-exchange`` 2.3.0 to
  3.0.0. In kit-less installs ``pxr`` is now OpenUSD **26.08** instead of 25.05. Environments
  that pin Isaac Lab against OpenUSD 25.05 outside Kit must re-resolve, and any code compiled
  or generated against the 25.05 ``pxr`` ABI must be rebuilt against 26.08. Kit-backed runs are
  unaffected: Kit serves its own OpenUSD from its extension roots.
* Excluded ``usd-optimize`` from kit-less installs. It is the only compiled OpenUSD consumer in
  the importer stack and has no 26.08 build, so it cannot load alongside ``usd-exchange`` 3.0.0.
  As a result ``UrdfConverterCfg.merge_mesh`` and ``MjcfConverterCfg.merge_mesh`` have no effect
  kit-less and log the miss; run the conversion under Kit if mesh merging is required. This is
  restored by removing the override once ``usd-optimize`` publishes an OpenUSD 26.08 build.
