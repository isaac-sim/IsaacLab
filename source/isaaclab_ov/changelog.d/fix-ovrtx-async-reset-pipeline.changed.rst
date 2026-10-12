* Changed :meth:`~isaaclab_ov.renderers.OVRTXRenderer.reset` to keep the asynchronous render pipeline on
  a partial environment reset. The previous re-prime made every step wait for its own capture when any
  environment reset, so asynchronous rendering gave no speedup in RL training. The reset environments
  now show their previous episode for one step. A full reset still primes a fresh image.
