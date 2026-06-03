Added
^^^^^

* Added ``ISAACLAB_PIN_KIT_GPU`` env var for :class:`~isaaclab.app.AppLauncher`.
  When set to a truthy value, appends ``--/renderer/multiGpu/enabled=False``,
  ``--/renderer/multiGpu/autoEnable=False`` and ``--/renderer/multiGpu/maxGpuCount=1``
  to the Kit command line so each Kit process touches only its assigned
  GPU (rather than enumerating every visible GPU at startup). Used by the
  multi-GPU CI workflow to prevent the shared cubric / PhysX-fabric
  GPU-interop context across sibling shards that surfaces as
  ``[Error] [omni.physx.plugin] Stage X already attached`` and
  ``SimulationApp.close`` hangs (see https://github.com/isaac-sim/IsaacLab/issues/3475
  and NVBug 5687364). Off by default; single-GPU and user-facing rendering
  paths are unchanged.
