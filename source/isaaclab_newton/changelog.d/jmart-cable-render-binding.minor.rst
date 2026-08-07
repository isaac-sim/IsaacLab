Added
^^^^^

* Added :meth:`~isaaclab_newton.physics.NewtonManager.collect_cable_segment_shapes`, which maps each
  renderable cable prim path to its ordered Newton segment shape ids. It reads only the Newton model
  and the USD stage, so kit-less renderers can drive cable points without a Fabric sync path.

Changed
^^^^^^^

* Changed :meth:`~isaaclab_newton.physics.NewtonManager.sync_cables_to_usd` to select its Fabric
  prims on the simulation device and run its kernel there, instead of mirroring the Newton model to
  the host and running on the CPU. The host mirror existed because the RTX Hydra render delegate
  could not read GPU-backed Fabric arrays for ``BasisCurves.points``; that gap is fixed upstream.
  This removes a device-to-host copy of the whole model's ``body_q`` on every dirty render frame,
  which scaled with total body count rather than with the number of cable segments actually read.

Fixed
^^^^^

* Fixed the Fabric cable sync silently doing nothing for a whole session when the Fabric stage was
  not yet available at ``start_simulation``. The stage handle was resolved exactly once and cached,
  so cables simulated correctly and rendered frozen at their spawn pose. The cable sync now
  re-acquires the stage on first use, and ``start_simulation`` warns instead of dereferencing a
  handle it does not have.
* Fixed cables rendering frozen under Kit-based renderers even when the Fabric write succeeded. The
  sync writes ``points`` in place from a Warp kernel, which leaves a render delegate's cached curve
  untouched, so each cable prim is now invalidated explicitly after the write.
* Fixed :class:`~isaaclab_newton.physics.NewtonManager` being unimportable inside a Kit session
  whose bundled Warp lags the one Newton's solvers require, by deferring a module-level
  ``newton.solvers`` import.
