* **Breaking:** Changed ``IsaacContrib-Lift-Soft-Franka-Custom-Coupling`` and its ``newton_mjwarp_vbd`` preset to
  use generic proxy coupling with a migration warning. Use ``Isaac-Lift-Soft-Franka`` with
  ``physics=newton_mjwarp_vbd_proxy`` for new runs, and retune policies that used the former manual solver.
