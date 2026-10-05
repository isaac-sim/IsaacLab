* Fixed the feet-wrench observation of the warp locomotion environment taking the feet in the joint-wrench sensor's
  own body order instead of the order of ``feet_body_names``, as the stable ``LocomotionDirectEnv`` now does. The
  published tasks run on Newton, where both orders agree, so their observation is unchanged.
