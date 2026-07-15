Fixed
^^^^^

* Fixed cross-test contamination in golden image tests when tiled-camera and viewport captures
  run sequentially in the same process.  Three sources were addressed: (1) the stale
  :class:`~isaaclab_newton.physics.NewtonManager` shadow model from the tiled stage persisting
  into the viewport test on the PhysX backend (cleared in the between-test prepare step);
  (2) CUDA RNG state drift causing the initial cartpole pole angle to differ between isolated
  and suite runs (seed is now applied immediately before ``env.reset()``); and (3) test ordering
  in both golden test files reordered to run tiled captures before viewport captures to prevent
  RTX render-product state from contaminating tiled camera output.  The ``newton-kit-viewport``
  golden image was regenerated to match the corrected deterministic initial physics state.
