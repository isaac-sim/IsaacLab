Fixed
^^^^^

* Fixed repeated OVPhysX articulation body-frame center-of-mass pose reads by caching them as model
  properties and invalidating dependent buffers when center-of-mass offsets are updated.
