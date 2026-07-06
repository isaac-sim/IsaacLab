Changed
^^^^^^^

* Changed the Disney DR Legs walk / hold-pose Kamino stepping to ``sim.dt = 1/150``
  with ``decimation = 3`` (previously ``0.004`` / ``5``). This keeps the 50 Hz control
  rate while reducing the number of Kamino solver calls per control step from 5 to 3,
  for roughly 1.6x faster simulation with equivalent task performance.
* Changed the ``newton_kamino`` presets of the velocity (A1, AnymalB, AnymalC, Go1, Go2,
  Cassie, G1, H1, Spot), cabinet, and shadow-hand reorient tasks to use the default
  single physics substep (removed the ``num_substeps=2`` override), and dropped the
  redundant explicit ``num_substeps`` overrides from the cartpole, ant, and reach Kamino
  presets.
