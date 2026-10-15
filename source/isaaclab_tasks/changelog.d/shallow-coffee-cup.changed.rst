* Added a shallower coffee cup with a circular handle and reduced its water fill.
* Increased physical coffee sampling density and position jitter while reducing
  the default passive rendering grains to one per particle.
* Kept the coffee seed volume fixed as its sampling density changed, and adapted
  the pour, stirring clearance and camera framing to the shorter cup.
* Reduced the spoon's collision expansion to preserve its shallow cavity, and
  deepened the pickup without changing the spoon geometry.
* Reduced the powder fill and shortened the motion with a nearby initial pose,
  continuous low transfer to a closer cup, and direct pouring-to-stirring motion.
* Refined the native liquid surface grid, smoothed its density field and limited
  anisotropic stretching independently of the physics resolution.
* Deepened the dissolved coffee's brown tint for the one-scoop dose.
* Raised the loaded spoon 15 mm higher and 4 mm farther from the robot before
  transferring, and kept the initial carry arc over the bowl to shed loose rim
  grains without lengthening the motion.
