* Changed the ``newton_kamino`` preset of the velocity tasks to Kamino's SCC solver with sparse
  Jacobian and dynamics and the ``LLTBRCM`` bilateral solver, in place of the dense P-ADMM
  solver. ``Isaac-Velocity-Rough-G1`` with 4096 environments steps about 50x faster.
