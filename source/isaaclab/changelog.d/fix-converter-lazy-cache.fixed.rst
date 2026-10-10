* Fixed lazy conversion in :class:`~isaaclab.sim.converters.MjcfConverter` and
  :class:`~isaaclab.sim.converters.UrdfConverter` loading the first output again after the asset file or the
  configuration changed in a reused ``usd_dir``. The Isaac Sim importers write each new conversion to a numbered
  folder such as ``<name>_1`` instead of overwriting ``<name>``, so ``.asset_hash`` now records the output of each
  asset and configuration in ``usd_dir``, and switching from configuration A to B and back to A reuses the first
  output. A record of an earlier version names only its last conversion. That output is still reused if it came from
  a lazy conversion, no numbered folder such as ``<name>_1`` lies next to it, and it is requested before anything else
  is converted into ``usd_dir``; other outputs are converted once more.
