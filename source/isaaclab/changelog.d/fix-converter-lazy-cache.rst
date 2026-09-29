Fixed
^^^^^

* Fixed lazy conversion in :class:`~isaaclab.sim.converters.MjcfConverter` and
  :class:`~isaaclab.sim.converters.UrdfConverter` loading the first output again after the asset file or the
  configuration changed in a reused ``usd_dir``. The Isaac Sim importers write each new conversion to a numbered
  folder such as ``<name>_1`` instead of overwriting ``<name>``, so the converters recorded the requested and the
  generated USD file next to the asset hash and loaded the generated one. Records written by earlier versions do
  not name these files, so each existing ``usd_dir`` was converted once more, by URDF and MJCF into a new numbered
  folder. A
  read-only ``usd_dir`` with such a record has to be converted again where it is writable.
