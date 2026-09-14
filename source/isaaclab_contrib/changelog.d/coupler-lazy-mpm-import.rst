Fixed
^^^^^

* Fixed the coupled-solver manager importing the implicit-MPM manager at module import time.
  That import pulls in ``warp.fem``, which resolves to the Warp bundled with Kit's
  ``omni.warp.core`` extension while ``warp._src`` resolves to the installed Warp, so it raised
  ``ImportError: cannot import name 'warn' from 'warp._src.utils'`` in any app that loads the
  extension. Coupled tasks with no MPM entry, such as an MJWarp and VBD proxy scene, could not be
  created under teleoperation. The import is now deferred and its call sites are guarded on
  whether the active configuration owns an MPM entry; with none they were already no-ops, so MPM
  behavior is unchanged.
